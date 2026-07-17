import numpy as np

class CohensD:
    @staticmethod
    def independent(x, y, pooled=True):
        x = np.asarray(x)
        y = np.asarray(y)
        nx = len(x)
        ny = len(y)
        mean_diff = np.mean(x) - np.mean(y)
        var_x = np.var(x, ddof=1)
        var_y = np.var(y, ddof=1)
        
        if pooled:
            sd_pooled = np.sqrt(((nx - 1) * var_x + (ny - 1) * var_y) / (nx + ny - 2))
            d = mean_diff / sd_pooled if sd_pooled != 0 else 0
        else:
            d = mean_diff / np.sqrt(var_x) if var_x != 0 else 0
            
        g = d * (1 - (3 / (4 * (nx + ny) - 9)))
        return {"d": d, "g": g}

    @staticmethod
    def paired(x, y):
        diff = np.asarray(x) - np.asarray(y)
        d_z = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff, ddof=1) != 0 else 0
        return d_z

    @staticmethod
    def interpret(d):
        d = abs(d)
        if d < 0.2: return "negligible"
        if d < 0.5: return "small"
        if d < 0.8: return "medium"
        if d < 1.2: return "large"
        return "very large"

class CohensF:
    pass

class ProportionEffectSizes:
    pass

class NonParametricEffectSizes:
    pass
