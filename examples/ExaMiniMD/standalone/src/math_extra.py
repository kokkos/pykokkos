#!/usr/bin/env python3

from typing import List
import math

class MathExtra:

    @classmethod
    def copy3(cls, v : List[float], ans : List[float]) -> None:
        ans[0] = v[0]
        ans[1] = v[1]
        ans[2] = v[2]

    @classmethod
    def zero3(cls, v : List[float]) -> None:
        v[0] = 0.0
        v[1] = 0.0
        v[2] = 0.0

    @classmethod
    def norm3(cls, v : List[float]) -> None:
        scale : float = 1.0 / math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
        v[0] *= scale
        v[1] *= scale
        v[2] *= scale

    @classmethod
    def normalize3(cls, v : List[float], ans : List[float]) -> None:
        scale : float = 1.0 / math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
        ans[0] = v[0]*scale
        ans[1] = v[1]*scale
        ans[2] = v[2]*scale

    @classmethod
    def snormalize3(cls, length : float, v : List[float], ans : List[float]) -> None:
        scale : float = length / math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
        ans[0] = v[0]*scale
        ans[1] = v[1]*scale
        ans[2] = v[2]*scale

    @classmethod
    def negate3(cls, v : List[float]) -> None:
        v[0] = -v[0]
        v[1] = -v[1]
        v[2] = -v[2]

    @classmethod
    def scale3(cls, s : float, v : List[float]) -> None:
        v[0] *= s
        v[1] *= s
        v[2] *= s

    @classmethod
    def add3(cls, v1 : List[float], v2 : List[float], ans : List[float]) -> None:
        ans[0] = v1[0] + v2[0]
        ans[1] = v1[1] + v2[1]
        ans[2] = v1[2] + v2[2]

    @classmethod
    def scaleadd3(cls, s : float, v1 : List[float], v2 : List[float], ans : List[float]) -> None:
        ans[0] = s*v1[0] + v2[0]
        ans[1] = s*v1[1] + v2[1]
        ans[2] = s*v1[2] + v2[2]

    @classmethod
    def sub3(cls, v1 : List[float], v2 : List[float], ans : List[float]) -> None:
        ans[0] = v1[0] - v2[0]
        ans[1] = v1[1] - v2[1]
        ans[2] = v1[2] - v2[2]

    @classmethod
    def len3(cls, v : List[float]) -> float:
        return math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])

    @classmethod
    def lensq3(cls, v : List[float]) -> float:
        return v[0]*v[0] + v[1]*v[1] + v[2]*v[2]

    @classmethod
    def distsq3(cls, v1 : List[float], v2 : List[float]) -> float:
        dx : float = v1[0] - v2[0]
        dy : float = v1[1] - v2[1]
        dz : float = v1[2] - v2[2]
        return dx*dx + dy*dy + dz*dz

    @classmethod
    def dot3(cls, v1 : List[float], v2 : List[float]) -> float:
        return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]

    @classmethod
    def cross3(cls, v1 : List[float], v2 : List[float], ans : List[float]) -> None:
        ans[0] = v1[1]*v2[2] - v1[2]*v2[1]
        ans[1] = v1[2]*v2[0] - v1[0]*v2[2]
        ans[2] = v1[0]*v2[1] - v1[1]*v2[0]

    @classmethod
    def col2mat(cls, ex : List[float], ey : List[float], ez : List[float], m : List[List[float]]) -> None:
        m[0][0] = ex[0]
        m[1][0] = ex[1]
        m[2][0] = ex[2]
        m[0][1] = ey[0]
        m[1][1] = ey[1]
        m[2][1] = ey[2]
        m[0][2] = ez[0]
        m[1][2] = ez[1]
        m[2][2] = ez[2]
        return

    @classmethod
    def det3(cls, m : List[List[float]]) -> float:
        ans : float = m[0][0]*m[1][1]*m[2][2] - m[0][0]*m[1][2]*m[2][1] - m[1][0]*m[0][1]*m[2][2] + m[1][0]*m[0][2]*m[2][1] + m[2][0]*m[0][1]*m[1][2] - m[2][0]*m[0][2]*m[1][1]
        return ans

    @classmethod
    def diag_times3(cls, d : List[float], m : List[List[float]], ans : List[List[float]]) -> None:
        ans[0][0] = d[0]*m[0][0]
        ans[0][1] = d[0]*m[0][1]
        ans[0][2] = d[0]*m[0][2]
        ans[1][0] = d[1]*m[1][0]
        ans[1][1] = d[1]*m[1][1]
        ans[1][2] = d[1]*m[1][2]
        ans[2][0] = d[2]*m[2][0]
        ans[2][1] = d[2]*m[2][1]
        ans[2][2] = d[2]*m[2][2]

    @classmethod
    def times3_diag(cls, m : List[List[float]], d : List[float], ans : List[List[float]]) -> None:
        ans[0][0] = m[0][0]*d[0]
        ans[0][1] = m[0][1]*d[1]
        ans[0][2] = m[0][2]*d[2]
        ans[1][0] = m[1][0]*d[0]
        ans[1][1] = m[1][1]*d[1]
        ans[1][2] = m[1][2]*d[2]
        ans[2][0] = m[2][0]*d[0]
        ans[2][1] = m[2][1]*d[1]
        ans[2][2] = m[2][2]*d[2]

    @classmethod
    def plus3(cls, m : List[List[float]], m2 : List[List[float]], ans : List[List[float]]) -> None:
        ans[0][0] = m[0][0]+m2[0][0]
        ans[0][1] = m[0][1]+m2[0][1]
        ans[0][2] = m[0][2]+m2[0][2]
        ans[1][0] = m[1][0]+m2[1][0]
        ans[1][1] = m[1][1]+m2[1][1]
        ans[1][2] = m[1][2]+m2[1][2]
        ans[2][0] = m[2][0]+m2[2][0]
        ans[2][1] = m[2][1]+m2[2][1]
        ans[2][2] = m[2][2]+m2[2][2]

    @classmethod
    def times3(cls, m : List[List[float]], m2 : List[List[float]], ans : List[List[float]]) -> None:
        ans[0][0] = m[0][0]*m2[0][0] + m[0][1]*m2[1][0] + m[0][2]*m2[2][0]
        ans[0][1] = m[0][0]*m2[0][1] + m[0][1]*m2[1][1] + m[0][2]*m2[2][1]
        ans[0][2] = m[0][0]*m2[0][2] + m[0][1]*m2[1][2] + m[0][2]*m2[2][2]
        ans[1][0] = m[1][0]*m2[0][0] + m[1][1]*m2[1][0] + m[1][2]*m2[2][0]
        ans[1][1] = m[1][0]*m2[0][1] + m[1][1]*m2[1][1] + m[1][2]*m2[2][1]
        ans[1][2] = m[1][0]*m2[0][2] + m[1][1]*m2[1][2] + m[1][2]*m2[2][2]
        ans[2][0] = m[2][0]*m2[0][0] + m[2][1]*m2[1][0] + m[2][2]*m2[2][0]
        ans[2][1] = m[2][0]*m2[0][1] + m[2][1]*m2[1][1] + m[2][2]*m2[2][1]
        ans[2][2] = m[2][0]*m2[0][2] + m[2][1]*m2[1][2] + m[2][2]*m2[2][2]

    @classmethod
    def transpose_times3(cls, m : List[List[float]], m2 : List[List[float]], ans : List[List[float]]) -> None:
        ans[0][0] = m[0][0]*m2[0][0] + m[1][0]*m2[1][0] + m[2][0]*m2[2][0]
        ans[0][1] = m[0][0]*m2[0][1] + m[1][0]*m2[1][1] + m[2][0]*m2[2][1]
        ans[0][2] = m[0][0]*m2[0][2] + m[1][0]*m2[1][2] + m[2][0]*m2[2][2]
        ans[1][0] = m[0][1]*m2[0][0] + m[1][1]*m2[1][0] + m[2][1]*m2[2][0]
        ans[1][1] = m[0][1]*m2[0][1] + m[1][1]*m2[1][1] + m[2][1]*m2[2][1]
        ans[1][2] = m[0][1]*m2[0][2] + m[1][1]*m2[1][2] + m[2][1]*m2[2][2]
        ans[2][0] = m[0][2]*m2[0][0] + m[1][2]*m2[1][0] + m[2][2]*m2[2][0]
        ans[2][1] = m[0][2]*m2[0][1] + m[1][2]*m2[1][1] + m[2][2]*m2[2][1]
        ans[2][2] = m[0][2]*m2[0][2] + m[1][2]*m2[1][2] + m[2][2]*m2[2][2]

    @classmethod
    def times3_transpose(cls, m : List[List[float]], m2 : List[List[float]], ans : List[List[float]]) -> None:
        ans[0][0] = m[0][0]*m2[0][0] + m[0][1]*m2[0][1] + m[0][2]*m2[0][2]
        ans[0][1] = m[0][0]*m2[1][0] + m[0][1]*m2[1][1] + m[0][2]*m2[1][2]
        ans[0][2] = m[0][0]*m2[2][0] + m[0][1]*m2[2][1] + m[0][2]*m2[2][2]
        ans[1][0] = m[1][0]*m2[0][0] + m[1][1]*m2[0][1] + m[1][2]*m2[0][2]
        ans[1][1] = m[1][0]*m2[1][0] + m[1][1]*m2[1][1] + m[1][2]*m2[1][2]
        ans[1][2] = m[1][0]*m2[2][0] + m[1][1]*m2[2][1] + m[1][2]*m2[2][2]
        ans[2][0] = m[2][0]*m2[0][0] + m[2][1]*m2[0][1] + m[2][2]*m2[0][2]
        ans[2][1] = m[2][0]*m2[1][0] + m[2][1]*m2[1][1] + m[2][2]*m2[1][2]
        ans[2][2] = m[2][0]*m2[2][0] + m[2][1]*m2[2][1] + m[2][2]*m2[2][2]

    @classmethod
    def invert3(cls, m : List[List[float]], ans : List[List[float]]) -> None:
        den : float = m[0][0]*m[1][1]*m[2][2]-m[0][0]*m[1][2]*m[2][1]
        den += -m[1][0]*m[0][1]*m[2][2]+m[1][0]*m[0][2]*m[2][1]
        den += m[2][0]*m[0][1]*m[1][2]-m[2][0]*m[0][2]*m[1][1]
        
        ans[0][0] = (m[1][1]*m[2][2]-m[1][2]*m[2][1]) / den
        ans[0][1] = -(m[0][1]*m[2][2]-m[0][2]*m[2][1]) / den
        ans[0][2] = (m[0][1]*m[1][2]-m[0][2]*m[1][1]) / den
        ans[1][0] = -(m[1][0]*m[2][2]-m[1][2]*m[2][0]) / den
        ans[1][1] = (m[0][0]*m[2][2]-m[0][2]*m[2][0]) / den
        ans[1][2] = -(m[0][0]*m[1][2]-m[0][2]*m[1][0]) / den
        ans[2][0] = (m[1][0]*m[2][1]-m[1][1]*m[2][0]) / den
        ans[2][1] = -(m[0][0]*m[2][1]-m[0][1]*m[2][0]) / den
        ans[2][2] = (m[0][0]*m[1][1]-m[0][1]*m[1][0]) / den

    @classmethod
    def matvec(cls, m : List[List[float]], v : List[float], ans : List[float]) -> None:
        ans[0] = m[0][0]*v[0] + m[0][1]*v[1] + m[0][2]*v[2]
        ans[1] = m[1][0]*v[0] + m[1][1]*v[1] + m[1][2]*v[2]
        ans[2] = m[2][0]*v[0] + m[2][1]*v[1] + m[2][2]*v[2]

    @classmethod
    def matvec(cls, ex : List[float], ey : List[float], ez : List[float], v : List[float], ans : List[float]) -> None:
        ans[0] = ex[0]*v[0] + ey[0]*v[1] + ez[0]*v[2]
        ans[1] = ex[1]*v[0] + ey[1]*v[1] + ez[1]*v[2]
        ans[2] = ex[2]*v[0] + ey[2]*v[1] + ez[2]*v[2]

    @classmethod
    def transpose_matvec(cls, m : List[List[float]], v : List[float], ans : List[float]) -> None:
        ans[0] = m[0][0]*v[0] + m[1][0]*v[1] + m[2][0]*v[2]
        ans[1] = m[0][1]*v[0] + m[1][1]*v[1] + m[2][1]*v[2]
        ans[2] = m[0][2]*v[0] + m[1][2]*v[1] + m[2][2]*v[2]

    @classmethod
    def transpose_matvec(cls, ex : List[float], ey : List[float], ez : List[float], v : List[float], ans : List[float]) -> None:
        ans[0] = ex[0]*v[0] + ex[1]*v[1] + ex[2]*v[2]
        ans[1] = ey[0]*v[0] + ey[1]*v[1] + ey[2]*v[2]
        ans[2] = ez[0]*v[0] + ez[1]*v[1] + ez[2]*v[2]

    @classmethod
    def transpose_diag3(cls, m : List[List[float]], d : List[float], ans : List[List[float]]) -> None:
        ans[0][0] = m[0][0]*d[0]
        ans[0][1] = m[1][0]*d[1]
        ans[0][2] = m[2][0]*d[2]
        ans[1][0] = m[0][1]*d[0]
        ans[1][1] = m[1][1]*d[1]
        ans[1][2] = m[2][1]*d[2]
        ans[2][0] = m[0][2]*d[0]
        ans[2][1] = m[1][2]*d[1]
        ans[2][2] = m[2][2]*d[2]

    @classmethod
    def vecmat(cls, v : List[float], m : List[List[float]], ans : List[float]) -> None:
        ans[0] = v[0]*m[0][0] + v[1]*m[1][0] + v[2]*m[2][0]
        ans[1] = v[0]*m[0][1] + v[1]*m[1][1] + v[2]*m[2][1]
        ans[2] = v[0]*m[0][2] + v[1]*m[1][2] + v[2]*m[2][2]

    @classmethod
    def scalar_times3(cls, f : float, m : List[List[float]]) -> None:
        m[0][0] *= f; m[0][1] *= f; m[0][2] *= f
        m[1][0] *= f; m[1][1] *= f; m[1][2] *= f
        m[2][0] *= f; m[2][1] *= f; m[2][2] *= f

    @classmethod
    def multiply_shape_shape(cls, one : List[float], two : List[float], ans : List[float]) -> None:
        ans[0] = one[0]*two[0]
        ans[1] = one[1]*two[1]
        ans[2] = one[2]*two[2]
        ans[3] = one[1]*two[3] + one[3]*two[2]
        ans[4] = one[0]*two[4] + one[5]*two[3] + one[4]*two[2]
        ans[5] = one[0]*two[5] + one[5]*two[1]

    @classmethod
    def qnormalize(cls, q : List[float]) -> None:
        norm : float = 1.0 / sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3])
        q[0] *= norm
        q[1] *= norm
        q[2] *= norm
        q[3] *= norm

    @classmethod
    def qconjugate(cls, q : List[float], qc : List[float]) -> None:
        qc[0] = q[0]
        qc[1] = -q[1]
        qc[2] = -q[2]
        qc[3] = -q[3]

    @classmethod
    def vecquat(cls, a : List[float], b : List[float], c : List[float]) -> None:
        c[0] = -a[0]*b[1] - a[1]*b[2] - a[2]*b[3]
        c[1] = b[0]*a[0] + a[1]*b[3] - a[2]*b[2]
        c[2] = b[0]*a[1] + a[2]*b[1] - a[0]*b[3]
        c[3] = b[0]*a[2] + a[0]*b[2] - a[1]*b[1]

    @classmethod
    def quatvec(cls, a : List[float], b : List[float], c : List[float]) -> None:
        c[0] = -a[1]*b[0] - a[2]*b[1] - a[3]*b[2]
        c[1] = a[0]*b[0] + a[2]*b[2] - a[3]*b[1]
        c[2] = a[0]*b[1] + a[3]*b[0] - a[1]*b[2]
        c[3] = a[0]*b[2] + a[1]*b[1] - a[2]*b[0]

    @classmethod
    def quatquat(cls, a : List[float], b : List[float], c : List[float]) -> None:
        c[0] = a[0]*b[0] - a[1]*b[1] - a[2]*b[2] - a[3]*b[3]
        c[1] = a[0]*b[1] + b[0]*a[1] + a[2]*b[3] - a[3]*b[2]
        c[2] = a[0]*b[2] + b[0]*a[2] + a[3]*b[1] - a[1]*b[3]
        c[3] = a[0]*b[3] + b[0]*a[3] + a[1]*b[2] - a[2]*b[1]

    @classmethod
    def invquatvec(cls, a : List[float], b : List[float], c : List[float]) -> None:
        c[0] = -a[1]*b[0] + a[0]*b[1] + a[3]*b[2] - a[2]*b[3]
        c[1] = -a[2]*b[0] - a[3]*b[1] + a[0]*b[2] + a[1]*b[3]
        c[2] = -a[3]*b[0] + a[2]*b[1] - a[1]*b[2] + a[0]*b[3]

    @classmethod
    def axisangle_to_quat(cls, v : List[float], angle : float, quat : List[float]) -> None:
        halfa : float = 0.5*angle
        sina : float = math.sin(halfa)
        quat[0] = math.cos(halfa)
        quat[1] = v[0]*sina
        quat[2] = v[1]*sina
        quat[3] = v[2]*sina

    @classmethod
    def rotation_generator_x(cls, m : List[List[float]], ans : List[List[float]]) -> None:
        ans[0][0] = 0
        ans[0][1] = -m[0][2]
        ans[0][2] = m[0][1]
        ans[1][0] = 0
        ans[1][1] = -m[1][2]
        ans[1][2] = m[1][1]
        ans[2][0] = 0
        ans[2][1] = -m[2][2]
        ans[2][2] = m[2][1]

    @classmethod
    def rotation_generator_y(cls, m : List[List[float]], ans : List[List[float]]) -> None:
        ans[0][0] = m[0][2]
        ans[0][1] = 0
        ans[0][2] = -m[0][0]
        ans[1][0] = m[1][2]
        ans[1][1] = 0
        ans[1][2] = -m[1][0]
        ans[2][0] = m[2][2]
        ans[2][1] = 0
        ans[2][2] = -m[2][0]

    @classmethod
    def rotation_generator_z(cls, m : List[List[float]], ans : List[List[float]]) -> None:
        ans[0][0] = -m[0][1]
        ans[0][1] = m[0][0]
        ans[0][2] = 0
        ans[1][0] = -m[1][1]
        ans[1][1] = m[1][0]
        ans[1][2] = 0
        ans[2][0] = -m[2][1]
        ans[2][1] = m[2][0]
        ans[2][2] = 0

if __name__ == "__main__":
    MathExtra.copy3([1, 2, 3], [4, 5, 6])
    MathExtra.norm3([1, 2, 3])
