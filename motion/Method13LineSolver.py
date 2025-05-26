import numpy as np
import math


class Method13LineSolver:
    @staticmethod
    def solve(route, theory_route, L=0):
        # 0: x定位误差xx，1:y定位误差yy，2:z定位误差zz
        # 垂直度误差: 3:xy 4:xz 5:yx 6:yz 7:zx 8:zy
        # 角度误差: 9:xa 10:xb 11:xc 12:ya 13:yb 14:yc 15:za 16:zb 17:zc
        error_ret = [0] * 18
        d12 = np.linalg.norm(route[1] - route[0])
        d13 = np.linalg.norm(route[2] - route[0])
        d45 = np.linalg.norm(route[4] - route[3])
        d46 = np.linalg.norm(route[5] - route[3])
        d78 = np.linalg.norm(route[7] - route[6])
        d79 = np.linalg.norm(route[8] - route[6])

        xb = (route[2][0] - route[0][0]) / d13
        xc = (route[0][0] - route[1][0]) / d12
        ya = (route[3][1] - route[5][1]) / d46
        yc = (route[4][1] - route[3][1]) / d45
        za = (route[8][2] - route[6][2]) / d79
        zb = (route[6][2] - route[7][2]) / d78

        lx = theory_route[0][0]
        ly = theory_route[3][1]
        lz = theory_route[6][2]
        xx = ((route[0][0] + route[1][0] + route[2][0] - 3 * lx) + (3 * L - d13) * xb + d12 * xc) / 3
        yy = ((route[3][1] + route[4][1] + route[5][1] - 3 * ly) + (3 * L - d46) * ya + d45 * yc) / 3
        zz = ((route[6][2] + route[7][2] + route[8][2] - 3 * lz) + (3 * L - d79) * za + d78 * zb) / 3

        x = lx
        y = ly
        z = lz

        A = [[0, 0, 1, 0, 0, 0, 0, -L],
             [1, 0, 0, 0, 0, 0, L, 0],
             [0, 0, 0, 0, 1, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 1, 0, 1, 0, 0, z - L],
             [1, 0, 0, 0, 0, 1, L - z, 0],
             [0, 1, 0, 1, 0, 0, y, 0]]

        L10 = pow(pow(route[9][0], 2) + pow(route[9][1], 2), 0.5)
        L11 = pow(pow(route[10][0], 2) + pow(route[10][2], 2), 0.5)
        L12 = pow(pow(route[11][1], 2) + pow(route[11][2], 2), 0.5)
        L13 = pow(pow(route[12][0], 2) + pow(route[12][1], 2) + pow(route[12][2], 2), 0.5)
        theta10x = theory_route[9][0] / L10
        theta10y = theory_route[9][1] / L10
        theta11x = theory_route[10][0] / L11
        theta11z = theory_route[10][2] / L11
        theta12y = theory_route[11][1] / L12
        theta12z = theory_route[11][2] / L12
        theta13x = theory_route[12][0] / L13
        theta13y = theory_route[12][1] / L13
        theta13z = theory_route[12][2] / L13

        B = [(L10 - math.sqrt(pow(theory_route[9][0], 2) + pow(theory_route[9][1], 2))) * theta10x
             - xx + L * xb + theory_route[9][1] * xc + theory_route[9][1] * yc,
             (L10 - math.sqrt(pow(theory_route[9][0], 2) + pow(theory_route[9][1], 2))) * theta10y
             - yy - L * ya - theory_route[9][0] * xc,
             (L11 - math.sqrt(pow(theory_route[10][0], 2) + pow(theory_route[10][2], 2))) * theta11x
             - xx + L * xb - theory_route[10][2] * xb + L * zb - theory_route[10][2] * zb,
             (L11 - math.sqrt(pow(theory_route[10][0], 2) + pow(theory_route[10][2], 2))) * theta11z
             - zz + theory_route[10][0] * xb,
             (L12 - math.sqrt(pow(theory_route[11][1], 2) + pow(theory_route[11][2], 2))) * theta12y
             - yy - L * ya + theory_route[11][2] * ya - L * za + theory_route[11][2] * za,
             (L12 - math.sqrt(pow(theory_route[11][1], 2) + pow(theory_route[11][2], 2))) * theta12z
             - zz - theory_route[11][1] * ya,
             (L13 - math.sqrt(
                 pow(theory_route[12][0], 2) + pow(theory_route[12][0], 2) + pow(theory_route[12][1], 2))) * theta13x
             - xx + L * xb - theory_route[12][2] * xb + L * zb - theory_route[12][2] * L * zb
             + theory_route[12][1] * (xc + yc),
             (L13 - math.sqrt(
                 pow(theory_route[12][1], 2) + pow(theory_route[12][1], 2) + pow(theory_route[12][1], 2))) * theta13y
             - yy - L * ya + theory_route[12][2] * ya - L * za + theory_route[12][2] * za - theory_route[12][0] * xc,
             (L13 - math.sqrt(
                 pow(theory_route[12][2], 2) + pow(theory_route[12][2], 2) + pow(theory_route[12][1], 2))) * theta13z
             - zz - theory_route[12][1] * ya + theory_route[12][0] * xb]

        # 解，残差平方和，秩，奇异值
        lst_result, residuals, rank, s = np.linalg.lstsq(A, B, rcond=None)
        xy, xz, yx, yz, zx, zy, xa, yb = lst_result
        error_ret = [xx, yy, zz, xy, xz, yx, yz, zx, zy, xa, xb, xc, ya, yb, yc, za, zb, 0]
        return error_ret
