class MergedOptimizers:
    """
        Can be used to Merge 2 optimizers into one.
        This is useful when you want to apply different optimizers to different parameters while the algorithm
        implementation only supports one optimizer.
    """
    def __init__(self, opt_list):
        self.opt_list = opt_list

    def step(self):
        for opt in self.opt_list:
            opt.step()

    def zero_grad(self):
        for opt in self.opt_list:
            opt.zero_grad()
