import os
import subprocess
import unittest



def set_env(set_env_var, cwd):
    subprocess.run(["rm", "-rf", "pk_cpp"], cwd=cwd)

    if set_env_var:
        os.environ["PK_LOOP_FUSE"] = "1"
    else:
        try:
            del os.environ["PK_LOOP_FUSE"]
        except Exception as e:
            pass


class TestLoopFusion(unittest.TestCase):
    
    def setUp(self):
       self.cwd = os.path.dirname(os.path.realpath(__file__))

    def run_test(self, test_num, range_iterations):
        '''
        This is the only way I could figure to capture kernel output - by running kernel as another process. 
        Redirect wasn't working for pyk kernels

        :param test_num: test number in loop_fusion_kernels.py
        :param range_terations: number of iterations the kernel performs in parallel
        '''
        set_env(False, self.cwd)
        result_vanilla = subprocess.run(["python", "loop_fusion_kernels.py", str(test_num), str(range_iterations)], cwd=self.cwd, capture_output=True, text=True)
        vanilla_out = result_vanilla.stdout

        # Again but with env variable
        set_env(True, self.cwd)
        result_fused = subprocess.run(["python", "loop_fusion_kernels.py", str(test_num), str(range_iterations)], cwd=self.cwd, capture_output=True, text=True)
        fused_out = result_fused.stdout

        self.assertEqual(vanilla_out, fused_out)

    def test_double_loop(self):
        self.run_test(0, 1)

    def test_simple_nested(self):
        self.run_test(1, 1)

    def test_nested_doubles(self):
        self.run_test(2, 1)

    def test_nested_triples(self):
        self.run_test(3, 1)

    def test_nested_triples_noprint(self):
        self.run_test(4, 1)

    def test_view_manip_inbetween(self):
        self.run_test(5, 1)

    def test_inner_scopes(self):
        self.run_test(6, 1)

    def test_nader_fusable(self):
        self.run_test(7, 1)


if __name__ == "__main__":
    unittest.main()
