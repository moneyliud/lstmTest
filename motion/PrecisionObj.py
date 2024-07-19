class PrecisionObj:
    def __init__(self):
        self.pre = 0.0
        self.fixed = False
        pass

    def __str__(self) -> str:
        return "{ pre: %s , fixed: %s }" % (self.pre, self.fixed)

    def __repr__(self) -> str:
        return "{ pre: %s , fixed: %s }" % (self.pre, self.fixed)
