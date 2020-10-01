import pytest
import curve as c
import numpy as np

def comp_array(array1, array2, tol=0.0000001):
    residuals = np.abs(array1 - array2)
    max_resi = np.max(residuals)
    return max_resi < tol

R3 = np.sqrt(3)
PI = np.pi
t_points     = np.array([[0,0],[1,0],[0,R3]])
t_rs         = np.array([R3,1,2])
t_midpoints  = np.array([[0,R3/2],[0.5,0],[0.5,R3/2]])
t_tm_vectors = np.array([[0,-1],[1,0],[-1/2,R3/2]])
t_nm_vectors = np.array([[-1,0],[0,-1],[R3/2,1/2]])
t_thetas     = np.array([-PI/2,0,2*PI/3])
t_phis       = np.array([PI/2,2*PI/3,PI*150/180])
t_A          = R3/2
t_G          = np.array([1/3, R3/3])
s_points     = np.array([[0,0],[1,0],[1,1],[0,1]])
s_rs         = np.array([1,1,1,1])
s_midpoints  = np.array([[0,0.5],[0.5,0],[1,0.5],[0.5,1]])
s_tm_vectors = np.array([[0,-1],[1,0],[0,1],[-1,0]])
s_nm_vectors = np.array([[-1,0],[0,-1],[1,0],[0,1]])
s_thetas     = np.array([-PI/2,0,PI/2,PI])
s_phis       = np.array([PI/2,PI/2,PI/2,PI/2])
s_A          = 1
s_G          = np.array([1/2, 1/2])


@pytest.mark.parametrize(('N','expected'),[
    (3, np.array([0, 1/3, 2/3])),
    (5, np.array([0, 1/5, 2/5, 3/5, 4/5]))
])
def test_generate_us(N, expected):
    res = c._generate_us(N)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('points','expected'),[
    (t_points, t_rs),
    (s_points, s_rs)
])
def test_calc_rs(points, expected):
    res = c._calc_rs(points)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('points','expected'),[
    (t_points, t_midpoints),
    (s_points, s_midpoints)
])
def test_calc_midpoints(points, expected):
    res = c._calc_midpoints(points)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('points','expected'),[
    (t_points, t_tm_vectors),
    (s_points, s_tm_vectors)
])
def test_calc_tm_vectors(points, expected):
    res = c._calc_tm_vectors(points)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('tm_vectors','expected'),[
    (t_tm_vectors, t_nm_vectors),
    (s_tm_vectors, s_nm_vectors)
])
def test_calc_nm_vectors(tm_vectors, expected):
    res = c._calc_nm_vectors(tm_vectors)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('tm_vectors','expected'),[
    (t_tm_vectors, t_thetas),
    (s_tm_vectors, s_thetas)
])
def test_calc_thetas(tm_vectors, expected):
    res = c._calc_thetas(tm_vectors)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('thetas','expected'),[
    (t_thetas, t_phis),
    (s_thetas, s_phis)
])
def test_calc_phis(thetas, expected):
    res = c._calc_phis(thetas)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('points','expected'),[
    (t_points, t_A),
    (s_points, s_A)
])
def test_calc_A(points, expected):
    res = c._calc_A(points)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('points', 'A','expected'),[
    (t_points, t_A, t_G),
    (s_points, s_A, s_G)
])
def test_calc_G(points, A, expected):
    res = c._calc_G(points, A)
    assert comp_array(res, expected)

"""
@pytest.mark.parametrize(('points','expected'),[
    (t_points, t_tm_vectors)
])
def test_calc_kais(rs, tans, expected):
    res = c._calc_tm_vectors(points)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('points','expected'),[
    (t_points, t_tm_vectors)
])
def test_calc_t_vectors(tm_vectors, coss, expected):
    res = c._calc_tm_vectors(points)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('points','expected'),[
    (t_points, t_tm_vectors)
])
def test_calc_psis(sins, rs, omega, expected):
    res = c._calc_tm_vectors(points)
    assert comp_array(res, expected)

@pytest.mark.parametrize(('points','expected'),[
    (t_points, t_tm_vectors)
])
def test_calc_ws(psis, coss, expected):
    res = c._calc_tm_vectors(points)
    assert comp_array(res, expected)
"""
