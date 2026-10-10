import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('gpu_bg', Path(__file__).with_name('gpu-bg.py'))
gpu_bg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_bg)


class AttributionTest(unittest.TestCase):
    def test_usage_before_creator_does_not_leak_to_previous_process(self):
        entries = [
            {'AppUsage': [], 'IOUserClientCreator': 'pid 10, Safari'},
            {'AppUsage': [{'accumulatedGPUTime': 100}, {'accumulatedGPUTime': 20}],
             'IOUserClientCreator': 'pid 20, Google Chrome He'},
            {'IOUserClientCreator': 'pid 30, WindowServer', 'AppUsage': [{'accumulatedGPUTime': 7}]},
        ]
        self.assertEqual(gpu_bg.usage_from_registry(entries),
                         {'Safari (10)': 0, 'Google Chrome He (20)': 120, 'WindowServer (30)': 7})

    def test_nested_clients_and_multiple_clients_per_process(self):
        entries = [{'IORegistryEntryChildren': [
            {'IOUserClientCreator': 'pid 20, Google Chrome He', 'AppUsage': [{'accumulatedGPUTime': 8}]},
            {'IOUserClientCreator': 'pid 20, Google Chrome He', 'AppUsage': [{'accumulatedGPUTime': 9}]},
            {'AppUsage': [{'accumulatedGPUTime': 500}]},
        ]}]
        self.assertEqual(gpu_bg.usage_from_registry(entries), {'Google Chrome He (20)': 17})


if __name__ == '__main__':
    unittest.main()
