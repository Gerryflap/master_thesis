class MergedOptimizers:
    def __init__(self, opt_list):
        self.opt_list = opt_list

    def step(self):
        for opt in self.opt_list:
            opt.step()

    def zero_grad(self):
        for opt in self.opt_list:
            opt.zero_grad()