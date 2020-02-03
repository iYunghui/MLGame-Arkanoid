"""The template of the main script of the machine learning process
"""

import arkanoid.communication as comm
from arkanoid.communication import SceneInfo, GameInstruction
import pickle
import numpy as np

def ml_loop():
	"""The main loop of the machine learning process

	This loop is run in a seperate process, and communicates with the game process.

	Note that the game process won't wait for the ml process to generate the
	GameInstrcution. It is possible that the frame of the GameInstruction
	is behind of the current frame in the game process. Try to decrease the fps
	to avoid this situation.
	"""

	# === Here is the execution order of the loop === #
	# 1. Put the initialization code here.
	
	# 2. Inform the game process that ml process is ready before start the loop.
	filename = "model.sav"
	load_model = pickle.load(open(filename, 'rb'))
	comm.ml_ready()
	# 3. Start an endless loop.
	while True:
		# 3.1. Receive the scene information sent from the game process.
		scene_info = comm.get_scene_info()
		
		# 3.2. If the game is over or passed, the game process will reset
		#      the scene immediately and send the scene information again.
		#      Therefore, receive the reset scene information.
		#      You can do proper actions, when the game is over or passed.
		if scene_info.status == SceneInfo.STATUS_GAME_OVER or \
			scene_info.status == SceneInfo.STATUS_GAME_PASS:
			scene_info = comm.get_scene_info()

		# 3.3. Put the code here to handle the scene information
		inp_temp = np.array([scene_info.ball[0], scene_info.ball[1], scene_info.platform[0]])
		input = inp_temp[np.newaxis, :]
		# 3.4. Send the instruction for this frame to the game process
		if load_model.predict(input) == 1:
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_RIGHT)
		elif load_model.predict(input) == -1:
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_LEFT)
		else:
			comm.send_instruction(scene_info.frame, GameInstruction.CMD_NONE)
