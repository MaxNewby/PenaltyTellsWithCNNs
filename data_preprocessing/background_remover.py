import subprocess
from pathlib import Path
from config import ROOT

class BackgroundRemoval:
    """Runs the Swift background remover over every raw image."""

    def __init__(self):
        self.swift_dir = Path(f'{ROOT}/data_preprocessing/swift')
        self.exe = self.swift_dir / 'bin' / 'background_remover'
        self.players_dir = Path(ROOT) / 'image_data' / 'Outfield_Players'

    def compile(self):
        '''
        Compiles and
        :return:
        '''
        self.exe.parent.mkdir(parents=True, exist_ok=True) #If bin file to put executable file in does not exist yet, make it.
        #Compile and run the swiftc code.
        subprocess.run(
            ['swiftc', '-O',
             str(self.swift_dir / 'background_remover' / 'remove_backgrounds.swift'),
             str(self.swift_dir / 'main.swift'),
             '-o', str(self.exe)],
            check=True,
        )

    def remove_backgrounds(self, recompile=False): #Function
        if recompile or not self.exe.exists():
            self.compile()

        # output isn't captured, so the [SUCCESS]/[FAILED] lines stream to console
        result = subprocess.run([str(self.exe), str(self.players_dir)])

        if result.returncode == 1:
            raise RuntimeError('background_remover was called with the wrong arguments')
        if result.returncode == 2:
            print('Some images failed. Check the [FAILED] lines above.')
        return result.returncode
