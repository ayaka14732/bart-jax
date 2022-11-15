import fire
from glob import iglob
import os

def test(tpu: bool=False) -> None:
    for filename in iglob('test/**/*.py'):
        print(f'Testing {filename}...')
        os.system(f'python "{filename}"')

    if tpu:
        for filename in iglob('test_tpu/**/*.py'):
            print(f'Testing {filename}...')
            os.system(f'python "{filename}"')

if __name__ == '__main__':
    fire.Fire(test)
