from pathlib import PurePath

from lab.lab import Lab

# Create a lab with the path of the project
lab = Lab(path=PurePath(__file__).parent)
