"""
python labml_samples/pytorch/ddp/launch.py --nproc_per_node=3 labml_samples/pytorch/ddp/mnist.py
"""
import os
import subprocess
import sys

from labml.logger import Text

from labml import experiment, logger


def main():
    if 'RUN_UUID' not in os.environ:
        os.environ['RUN_UUID'] = experiment.generate_uuid()

    logger.log(str(sys.argv), Text.danger)
    cmd = [sys.executable, '-u', '-m', 'torch.distributed.launch', *sys.argv[1:]]
    # print(cmd)
    try:
        process = subprocess.Popen(cmd, env=os.environ)
        # print('wait')
        process.wait()
    except Exception as e:
        logger.log('Error starting launcher', Text.danger)
        raise e

    if process.returncode != 0:
        logger.log('Launcher failed', Text.danger)
        raise subprocess.CalledProcessError(returncode=process.returncode, cmd=cmd)


if __name__ == '__main__':
    main()
