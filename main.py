import sys
import vpg_quadball
import spg_quadball
import quad_ppo

if __name__ =="__main__":
	# pg train 100 1
	print(sys.argv)
	if not len(sys.argv) <= 5:
		print("Invalid number of arguments provided. Try again..")
		sys.exit()
	algorithm = sys.argv[1]
	action = sys.argv[2]
	try:
		epochs = int(sys.argv[3])
	except:
		epochs = 200
	try:
		runs_per_epoch = int(sys.argv[4])
	except:
		runs_per_epoch = 50
	
	

	if algorithm == 'pg':
		algo = spg_quadball.SimplePolicyGradient(epochs=epochs,runs_per_epoch=runs_per_epoch)
	elif algorithm == 'vpg':
		algo = vpg_quadball.QuadBounceBallVPG(total_episodes=epochs,steps_per_episode=runs_per_epoch)
	elif algorithm == 'ppo':
		algo = quad_ppo.ProximalPolicy(epochs=epochs,epoch_runs=runs_per_epoch)
	else:
		print("Invalid Algorithm chosen")

	if action == "eval":
		algo.eval()
	elif action == "train":
		algo.train()
	else:
		print("Invalid action chosen")