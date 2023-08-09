import datetime
import gc
import pathlib

import torch
from torch.utils.benchmark import Timer
# from torch.profiler import _memory_profiler
# from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline

torch.backends.cuda.matmul.allow_tf32 = True

BATCH_SIZE = 2048
FEATURE_DIM = 8 * 1024
NUM_LAYERS = 20

torch.cuda.set_device(1)




def set_adamw_hook(mod, p, lr, betas):
    """Based on https://gist.github.com/albanD/18c240bd2e09f9d93f5c4a0c9ccda39e"""
    acc_grad = p.view_as(p).grad_fn.next_functions[0][0]

    # The grad accumulator is a weak ref, so we need to keep it
    # alive until the Tensor is alive.
    # Store it on the module to avoid uncollectable ref-cycle
    if not hasattr(mod, "_acc_grads"):
        mod._acc_grads = []
    mod._acc_grads.append(acc_grad)

    optimizer = torch.optim.AdamW([p], lr=lr, betas=betas, foreach=False)
    def adamw_hook(*_unused):
        optimizer.step()
        optimizer.zero_grad()

    # We should have an API for post hooks... But we don't have one right now
    acc_grad.register_hook(adamw_hook)


def main():
    trace_dir = pathlib.Path(__file__).parent.joinpath("traces")
    trace_dir.mkdir(exist_ok=True)
    now = datetime.datetime.now().strftime("%Y_%m_%d:%H.%M.%S")

    for opt in ("backward_hook", "foreach", "default"):
        backbone = [
            torch.nn.Linear(FEATURE_DIM, FEATURE_DIM, bias=False)
            for _ in range(NUM_LAYERS)
        ]
        model = torch.nn.Sequential(
            *backbone,
            torch.nn.Linear(FEATURE_DIM, 1)
        )
        model.cuda()

        if opt == "backward_hook":
            for p in model.parameters():
                set_adamw_hook(model, p, lr=.01, betas=(0.1, 0.1))

            def optimizer_step():
                pass

        else:
            assert opt in ("foreach", "default")
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=0.01,
                betas=(0.1, 0.1),
                foreach=(opt == "foreach")
            )

            def optimizer_step():
                optimizer.step()
                optimizer.zero_grad()

        def step():
            x = torch.ones((BATCH_SIZE, FEATURE_DIM), device="cuda")
            model(x).mean().backward()
            optimizer_step()

        # This also serves to warm up the model and ensure that we are in a steady state.
        print(Timer("step()", globals={"step": step}).blocked_autorange(min_run_time=1))

        gc.collect()
        with torch.profiler.profile(record_shapes=True, profile_memory=True, with_stack=True) as p:
            step()
            gc.collect()

        p.export_memory_timeline(str(trace_dir.joinpath(f"linear_stack_{now}_{opt=}.html")), torch.cuda.current_device())

        # Alternate method of rendering the memory. (Per Tensor.)
        # plot_html = torch.cuda._memory_viz.profile_plot(p, device=torch.device(f"cuda:0"))
        # with open(trace_dir.joinpath(f"linear_stack_{now}_{opt=}.html"), "wt") as f:
        #     f.write(plot_html)


if __name__ == "__main__":
    main()