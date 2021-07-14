
## Instrucciones para ejecucion
Asumiendo que el usuario esta corriendo linux o idealmente ubuntu con python instalado, y gym retro descargado y extraido localmente de https://github.com/openai/retro/releases/tag/v0.8.0 corra lo siguiente en la carpeta donde descargo
>cd retro-0.8.0
>pip3 install -e .

Obten Legalmente una ROM de Super Mario World, e importala a retro
>python3 -m retro.import /"DirectorioDondeGuardoLaROM"
>pip3 install neat-python
>pip3 install numpy
>pip3 install graphviz
>pip3 install matplotlib
>pip3 install opencv-python

finalmente con el repositorio del proyecto descargado en 

>/Controler/custom_integration/smw

se copian los contenidos de esta carpeta a

>/gym-retro/retro/data/stable/SuperMarioWorld-Snes"

para correr el algoritmo principal en la carpeta controler/chimera
>python3 chimeraMK1.py
