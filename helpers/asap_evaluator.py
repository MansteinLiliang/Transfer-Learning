from __future__ import absolute_import
from __future__ import print_function
from scipy.stats import pearsonr, spearmanr, kendalltau
import logging
import numpy as np
from .my_kappa_calculator import quadratic_weighted_kappa as qwk
from .my_kappa_calculator import linear_weighted_kappa as lwk
import theano
from . import data_utils as U

class Evaluator():
	def __init__(self, dataset, prompt_id, out_dir, dev_y_org, test_y_org, batch_size=128):
		out_dir = './out_dir/prompt_' + str(prompt_id)
		self.logger = logging.getLogger(__name__)
		U.set_logger(self.logger, out_dir)
		self.dev_y_org = dev_y_org
		self.test_y_org = test_y_org
		self.dataset = dataset
		self.prompt_id = prompt_id
		# self.out_dir = out_dir
		self.best_dev = [-1, -1, -1, -1]
		self.best_test = [-1, -1, -1, -1]
		self.best_dev_epoch = 0
		self.best_test_missed = 0
		self.best_test_missed_epoch = 0
		self.batch_size = batch_size
		self.low, self.high = self.dataset.get_score_range(self.prompt_id)

	def calc_correl(self, dev_pred, test_pred):
		dev_prs, _ = pearsonr(dev_pred, self.dev_y_org)
		test_prs, _ = pearsonr(test_pred, self.test_y_org)
		dev_spr, _ = spearmanr(dev_pred, self.dev_y_org)
		test_spr, _ = spearmanr(test_pred, self.test_y_org)
		dev_tau, _ = kendalltau(dev_pred, self.dev_y_org)
		test_tau, _ = kendalltau(test_pred, self.test_y_org)
		return dev_prs, test_prs, dev_spr, test_spr, dev_tau, test_tau
		# self.dump_ref_scores()

	def calc_qwk(self, dev_pred, test_pred):
		# Kappa only supports integer values
		dev_pred_int = np.rint(dev_pred).astype('int32')
		test_pred_int = np.rint(test_pred).astype('int32')
		dev_qwk = qwk(self.dev_y_org, dev_pred_int, self.low, self.high)
		test_qwk = qwk(self.test_y_org, test_pred_int, self.low, self.high)
		dev_lwk = lwk(self.dev_y_org, dev_pred_int, self.low, self.high)
		test_lwk = lwk(self.test_y_org, test_pred_int, self.low, self.high)
		return dev_qwk, test_qwk, dev_lwk, test_lwk

	def calc_dev_qwk(self, dev_pred):
		# Kappa only supports integer values
		dev_pred_int = np.rint(dev_pred).astype('int32')
		dev_qwk = qwk(self.dev_y_org, dev_pred_int, self.low, self.high)
		dev_lwk = lwk(self.dev_y_org, dev_pred_int, self.low, self.high)
		return dev_qwk, dev_lwk

	def calc_test_qwk(self, test_pred):
		# Kappa only supports integer values
		test_pred_int = np.rint(test_pred).astype('int32')
		test_qwk = qwk(self.test_y_org, test_pred_int, self.low, self.high)
		test_lwk = lwk(self.test_y_org, test_pred_int, self.low, self.high)
		return test_qwk, test_lwk

	def predict(self, model, x, masks, y, batch_size, feats=None):
		my_batch = self.dataset.dev_test_batch_generator(x, masks, y, batch_size, feats=feats)
		y_pred = []
		total_cost = 0.0
		count = 0.0
		for items in my_batch:
			if feats is None:
				(x_, mask, y_) = items
				true_cost, pred = model.predict(x_, np.asarray(mask, dtype=theano.config.floatX), y_, len(y_))
			else:
				(x_, feat, mask, y_) = items
				true_cost, pred = model.predict(x_, feat, np.asarray(mask, dtype=theano.config.floatX), y_, len(y_))
			y_pred.extend(pred.flatten())
			total_cost += true_cost
			count += 1.0
		return np.array(y_pred), total_cost / count

	def feature_evaluate(self, dev_pred, test_pred=None):
		logger = self.logger
		self.dev_pred = self.dataset.convert_to_dataset_friendly_scores(dev_pred, self.prompt_id)
		if test_pred is not None:
			self.test_pred = self.dataset.convert_to_dataset_friendly_scores(test_pred, self.prompt_id)
			self.dev_qwk, self.test_qwk, self.dev_lwk, self.test_lwk = self.calc_qwk(self.dev_pred, self.test_pred)
			logger.info(
				'[DEV]   QWK:  %.3f, LWK: %.3f, ',
				self.dev_qwk, self.dev_lwk
			)
			logger.info(
				'[TEST]  QWK:  %.3f, LWK: %.3f',
				self.test_qwk, self.test_lwk
			)
			return (self.dev_qwk, self.test_qwk)
		else:
			self.dev_qwk, self.dev_lwk = self.calc_dev_qwk(self.dev_pred)
			logger.info(
				'[DEV]   QWK:  %.3f, LWK: %.3f, ',
				self.dev_qwk, self.dev_lwk
			)
			return (self.dev_qwk,)

	def evaluate(self, dev_x, dev_masks, dev_y, test_x, test_masks, test_y, model, epoch, print_info=True, dev_feats=None, test_feats=None):
		self.dev_mean, self.test_mean, self.dev_std, self.test_std = np.array(dev_y).mean(), np.array(
			test_y).mean(), np.array(dev_y).std(), np.array(test_y).std()

		self.dev_pred, self.dev_loss = self.predict(model, dev_x, dev_masks, dev_y, self.batch_size, dev_feats)
		self.test_pred, self.test_loss = self.predict(model, test_x, test_masks, test_y, self.batch_size, test_feats)

		self.dev_pred = self.dataset.convert_to_dataset_friendly_scores(self.dev_pred, self.prompt_id)
		self.test_pred = self.dataset.convert_to_dataset_friendly_scores(self.test_pred, self.prompt_id)

		# print 'prediction example...'
		# self.dump_predictions(self.dev_pred, self.test_pred, epoch)

		self.dev_prs, self.test_prs, self.dev_spr, self.test_spr, self.dev_tau, self.test_tau = self.calc_correl(
			self.dev_pred, self.test_pred)

		self.dev_qwk, self.test_qwk, self.dev_lwk, self.test_lwk = self.calc_qwk(self.dev_pred, self.test_pred)
		if self.dev_qwk > self.best_dev[0]:
			self.best_dev = [self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_spr, self.dev_tau]
			self.best_test = [self.test_qwk, self.test_lwk, self.test_prs, self.test_spr, self.test_tau]
			self.best_dev_epoch = epoch
		# model.save_weights(self.out_dir + '/best_model_weights.h5', overwrite=True)

		if self.test_qwk > self.best_test_missed:
			self.best_test_missed = self.test_qwk
			self.best_test_missed_epoch = epoch

		if print_info:
			self.print_info(epoch)
		return


	def print_info(self, epoch):
		logger = self.logger
		logger.info("Current evaluation epoch step = (%d)" % epoch)
		logger.info('[Dev]   loss: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
			self.dev_loss, self.dev_pred.mean(), self.dev_mean, self.dev_pred.std(), self.dev_std))
		logger.info('[Test]  loss: %.4f, mean: %.3f (%.3f), stdev: %.3f (%.3f)' % (
			self.test_loss, self.test_pred.mean(), self.test_mean, self.test_pred.std(), self.test_std))
		logger.info(
			'[DEV]   QWK:  %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (
				self.dev_qwk, self.dev_lwk, self.dev_prs, self.dev_spr, self.dev_tau, self.best_dev_epoch,
				self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))
		logger.info(
			'[TEST]  QWK:  %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f (Best @ %i: {{%.3f}}, %.3f, %.3f, %.3f, %.3f)' % (
				self.test_qwk, self.test_lwk, self.test_prs, self.test_spr, self.test_tau, self.best_dev_epoch,
				self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[4]))

		logger.info(
			'--------------------------------------------------------------------------------------------------------------------------')
		return


	def print_final_info(self):
		logger = self.logger
		logger.info(
			'--------------------------------------------------------------------------------------------------------------------------')
		logger.info('Missed @ Epoch %i:' % self.best_test_missed_epoch)
		logger.info('  [TEST] QWK: %.3f' % self.best_test_missed)
		logger.info('Best @ Epoch %i:' % self.best_dev_epoch)
		logger.info('  [DEV]  QWK: %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f' % (
			self.best_dev[0], self.best_dev[1], self.best_dev[2], self.best_dev[3], self.best_dev[4]))
		logger.info('  [TEST] QWK: %.3f, LWK: %.3f, PRS: %.3f, SPR: %.3f, Tau: %.3f' % (
			self.best_test[0], self.best_test[1], self.best_test[2], self.best_test[3], self.best_test[4]))
