import cPickle as cp
import numpy as np
import h5py


def main():
	data = cp.load(open('/home/sanchit/Desktop/Semester 5/AI/Project/EEG Data/data_preprocessed_python/s01.dat', 'rb'))
	x=data['data']
	y=data['labels']

	participants=4
	videos=x.shape[0]
	channels=x.shape[1]
	readings=x.shape[2]
	labels=y.shape[1]

	print videos,channels,readings,labels
	print y.shape

	X=np.zeros((participants*videos,channels*(readings/32)))
	Y_1=np.zeros((participants*videos))
	Y_2=np.zeros((participants*videos))
	Y_3=np.zeros((participants*videos))
	Y_4=np.zeros((participants*videos))

	print X.shape
	print Y_1.shape

	var=0
	var2=0
	for participant in range(participants):
		if participant+1<10:
			num="0"+str(participant+1)
		else:
			num=str(participant+1)
		print num
		data = cp.load(open('/home/sanchit/Desktop/Semester 5/AI/Project/EEG Data/data_preprocessed_python/s'+num+'.dat', 'rb'))
		x=data['data']
		y=data['labels']
		for video in range(videos):
			var3=0
			for channel in range(channels):
				for reading in range(readings):
					if reading % 32 == 0:
						X[var][var3]=x[video][channel][reading]
						var3+=1
			# var2+=1

			Y_1[var]=y[video][0]
			Y_2[var]=y[video][1]
			Y_3[var]=y[video][2]
			Y_4[var]=y[video][3]
			var+=1

	for i in Y_1:
		print i
	print X.shape
	print Y_1.shape

	h5f = h5py.File('xdata.h5', 'w')
	h5f.create_dataset('dataset_1', data=X)
	h5f.close()

	h5f = h5py.File('y1data.h5', 'w')
	h5f.create_dataset('dataset_1', data=Y_1)
	h5f.close()

	h5f = h5py.File('y2data.h5', 'w')
	h5f.create_dataset('dataset_1', data=Y_2)
	h5f.close()

	h5f = h5py.File('y3data.h5', 'w')
	h5f.create_dataset('dataset_1', data=Y_3)
	h5f.close()

	h5f = h5py.File('y4data.h5', 'w')
	h5f.create_dataset('dataset_1', data=Y_4)
	h5f.close()


main()