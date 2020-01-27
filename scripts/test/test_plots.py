import rosbag
import numpy as np
import matplotlib.pyplot as pl
import os

from stat import S_ISREG, ST_CTIME, ST_MODE
import pickle

class TestPlots():
    def __init__(self, bagfull, bagname, plotpath):
        self.error_time = np.empty([0,1])
        self.errors = np.empty([0,1])
        self.train_time = np.empty([0,1])
        self.num_models = np.empty([0,1])
        self.njit = []
        self.wgen = []
        self.num_inducing = []
        self.degrees = []
        self.filename = bagfull
        self.bagname = bagname
        self.save_path = plotpath
        self.results = []
        self.mse = []
        self.rmse = []

    def extract_bag(self):
        bag = rosbag.Bag(self.filename)
        for msg in bag.read_messages(topics=['results']):
            self.results = msg.message
        bag.close()
        self.get_errors()
        self.get_params()
        self.get_updates()
        self.mse = np.mean(self.errors)
        self.rmse = np.sqrt(self.mse)
        
    def get_errors(self):
        for error in self.results.errors:
            self.errors = np.vstack((self.errors, error.data))
            self.error_time = np.vstack((self.error_time, error.header.stamp.to_sec()))

    def get_params(self):
        self.njit = self.results.params.njit
        self.num_inducing = self.results.params.inducing
        self.degrees = self.results.params.degrees
        self.wgen = self.results.params.wgen

    def get_updates(self):
        for upd in self.results.updates:
            self.train_time = np.vstack((self.train_time, upd.header.stamp.to_sec()))
            self.num_models = np.vstack((self.num_models, upd.data))
            

    def make_plot(self):

        # shift = np.min([self.error_time[0],self.train_time[0]])
        # print(np.size(self.error_time))
        shift = self.error_time[0]
        pl.plot(self.error_time - shift, self.errors)
        pl.vlines(self.train_time - shift, 0, np.max(self.errors), colors = 'g', linestyles = 'dotted' )

        # A = np.where(self.num_models[:-1] != self.num_models[1:])[0]
        A = np.where(np.roll(self.num_models,1)!=self.num_models)[0]
        if len(A) > 0:
            A = A[1:]

        pl.vlines((self.train_time - shift)[A], 0, np.max(self.errors), colors = 'r', linestyles = 'dotted' )

        pl.ylim(0, np.max(self.errors))
        pl.xlabel('Time [s]')
        pl.ylabel('squared error')
        # pl.text(15, 0.0004, 'rmse: ' + str(round(self.wgen, 4)))

        if len(self.num_models) > 0:
            textstr = '\n'.join((
                r'$\mathrm{inducing\_points}=%u$' % (self.num_inducing, ),
                r'$\mathrm{wgen}=%.3f$' % (self.wgen, ),
                r'$\mathrm{num\_models}=%u$' % (np.max(self.num_models), ),
                r'$\mathrm{RMSE}=%.4f$' % (self.rmse, )))
        else:
            textstr = '\n'.join((
                r'$\mathrm{inducing\_points}=%u$' % (self.num_inducing, ),
                r'$\mathrm{wgen}=%.3f$' % (self.wgen, ),
                r'$\mathrm{RMSE}=%.4f$' % (self.rmse, )))            

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # place a text box in upper left in axes coords
        pl.text(15, np.max(self.errors)*0.75, textstr, fontsize=12,
                verticalalignment='top', bbox=props)

        # pl.show()
        pl.savefig(self.save_path + self.bagname[:-4])
        pl.cla()

# rootdir = os.getcwd()
# imdir = os.path.join(rootdir, 'Figures')
dir_path = '/home/bpwilcox/catkin_ws/src/SOLAR_GP-ROS/bags/tests/'
plot_path = '/home/bpwilcox/catkin_ws/src/SOLAR_GP-ROS/plots/'
# dir_path = imdir   
data = (os.path.join(dir_path, fn) for fn in os.listdir(dir_path))
data = ((os.stat(path), path) for path in data)    
data = ((stat[ST_CTIME], path)
        for stat, path in data if S_ISREG(stat[ST_MODE]))

Results = dict()
Data = dict()
Data2 = dict()
# Data3 = dict()


# Data3['first'] = []
# Data3['redo'] = []
# Data3['no_train'] = []

num_inducing = range(15,50,5)
w_gen = np.linspace(0.8, 0.985, 5)

for n in num_inducing:
    Data[n] = []

# for w in w_gen:
#     Data2[str(round(w,2))] = []

for cdate, path in sorted(data):       
    file = os.path.join(dir_path,path)
    split_path = os.path.split(path)
    print(split_path[1])
    # T = TestPlots(directory + 'test_20190314-212323.bag')
    T = TestPlots(file, split_path[1], plot_path)
    T.extract_bag()
    T.make_plot()
    # Results[split_path[1]] = T.rmse
    # Data[T.num_inducing].append(T.rmse)
    # if str(round(T.wgen,3)) not in Data2:
    #     Data2[str(round(T.wgen,3))] = [T.rmse]
    # else:
    #     Data2[str(round(T.wgen,3))].append(T.rmse)

    # name = split_path[1]
    # if name[:name.rfind('_')] == 'test_'
# num_ind = []
# min_rmse = []
# avg_rmse = []

# for key, value in Data.iteritems():
#     num_ind.append(key)
#     # print(len(value))
#     min_rmse.append(np.min(value))
#     avg_rmse.append(np.mean(value))


# labels = ['M = 8', 'M = 3', 'M = 2', 'M = 2', 'M = 2', 'M = 8', 'M = 1']
# ind = np.argsort(num_ind)

# # num_ind = num_ind[ind]
# # min_rmse = min_rmse[ind]

# new_ind = []
# new_rmse = []
# for i in range(0,len(num_ind)):
#     new_ind.append(num_ind[ind[i]])
#     new_rmse.append(min_rmse[ind[i]])

# pl.bar(new_ind, new_rmse, width = 3.0)
# pl.xticks(new_ind, new_ind)
# pl.xlabel('Number of inducing points')
# pl.ylabel('Minimum RMSE')
# for i, r in enumerate(new_ind):
#     pl.text(x = r-1.25 , y = new_rmse[i] + 0.0001, s = labels[i], size = 8)

# # pl.show()
# pl.savefig(plot_path + 'min_ind')
# pl.cla()

# pl.bar(num_ind, avg_rmse, width = 3.0)
# pl.xticks(num_ind, num_ind)
# pl.xlabel('Number of inducing points')
# pl.ylabel('Average RMSE')
# # pl.show()
# pl.savefig(plot_path + 'avg_ind')
# pl.cla()


# num_ind = []
# min_rmse = []
# avg_rmse = []


# for key, value in Data2.iteritems():
#     num_ind.append(key)
#     # print(len(value))
#     min_rmse.append(np.min(value))
#     avg_rmse.append(np.mean(value))

# pl.bar(num_ind, min_rmse, width = 0.3)
# pl.xticks(num_ind, num_ind)
# pl.xlabel('Distance Metric Threshold (wgen)')
# pl.ylabel('Minimum RMSE')
# # pl.show()
# pl.savefig(plot_path + 'min_wgen')
# pl.cla()

# pl.bar(num_ind, avg_rmse, width = 0.3)
# pl.xticks(num_ind, num_ind)
# pl.xlabel('Distance Metric Threshold (wgen)')
# pl.ylabel('Average RMSE')
# # pl.show()
# pl.savefig(plot_path + 'avg_wgen')
# pl.cla()

# f = open("SimResults.pkl","wb")
# pickle.dump(Results,f)
# f.close()