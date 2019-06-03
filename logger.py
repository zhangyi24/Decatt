import logging
import sys
import os
import datetime

def create_logger(dataset):
	logger = logging.getLogger(name=dataset)
	logger.setLevel(logging.DEBUG)
	fmt = logging.Formatter('%(asctime)s %(message)s')
	logfile = get_log_filename(dataset)
	fh = logging.FileHandler(filename=logfile, mode='w', encoding='utf-8')
	fh.setLevel(logging.DEBUG)
	fh.setFormatter(fmt)
	sh = logging.StreamHandler(stream=sys.stdout)
	sh.setLevel(logging.DEBUG)
	sh.setFormatter(fmt)
	logger.addHandler(fh)
	logger.addHandler(sh)
	return logger

def get_log_filename(dataset):
	dir = create_or_get_dir(dataset)
	time = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
	filename = dataset + '_' + time + '.log'
	filename = os.path.join(dir, filename)
	return filename
	
def create_or_get_dir(dir):
	dir = os.path.join('log', dir)
	if not os.path.exists(dir):
		os.mkdir(dir)
	return dir
	
if __name__ == '__main__':
	logger = create_logger('quora')
	logger.info('1')
	logger.info('2')
	logger.info('3')