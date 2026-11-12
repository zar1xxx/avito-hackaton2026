import unittest

from src.common.resources import memory_snapshot


class ResourceTests(unittest.TestCase):
    def test_own_process_memory_is_recorded(self):
        result=memory_snapshot()
        self.assertGreater(result['peak_working_set_bytes'],0)
        if 'total_physical_bytes' in result:
            self.assertGreater(result['available_physical_bytes'],0)
            self.assertLessEqual(result['available_physical_bytes'],result['total_physical_bytes'])
