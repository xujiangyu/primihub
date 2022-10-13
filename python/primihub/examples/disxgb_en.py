#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 Copyright 2022 Primihub

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """
import primihub as ph
from primihub import dataset, context
from phe import paillier
from primihub.primitive.opt_paillier_c2py_warpper import *
# from primihub.channel.zmq_channel import IOService, Session
# from primihub.FL.proxy.proxy import ServerChannelProxy
# from primihub.FL.proxy.proxy import ClientChannelProxy
# from primihub.FL.model.xgboost.xgb_guest_en import XGB_GUEST_EN
# from primihub.FL.model.xgboost.xgb_host_en import XGB_HOST_EN
# from primihub.FL.model.xgboost.xgb_guest import XGB_GUEST
# from primihub.FL.model.xgboost.xgb_host import XGB_HOST
from primihub.FL.model.evaluation.evaluation import Regression_eva
from primihub.FL.model.evaluation.evaluation import Classification_eva
import pandas as pd
import numpy as np
import logging
import pickle

from primihub.primitive.opt_paillier_c2py_warpper import *
import time
import pandas as pd
import numpy as np
import copy
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from primihub.channel.zmq_channel import IOService, Session
import functools


LOG_FORMAT = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
logging.basicConfig(level=logging.DEBUG,
                    format=LOG_FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger("proxy")


def phe_map_enc(pub, pri, item):
    # return pub.encrypt(item)
    return opt_paillier_encrypt_crt(pub, pri, item)


def phe_map_dec(pub, prv, item):
    # return pri.decrypt(item)
    return opt_paillier_decrypt_crt(pub, prv, item)


def phe_add(enc1, enc2):
    return enc1 + enc2


# def opt_paillier_add(pub, x, y):
#     return opt_paillier_add(pub, x, y)


class ClientChannelProxy:
    def __init__(self, host, port, dest_role="NotSetYet"):
        self.ios_ = IOService()
        self.sess_ = Session(self.ios_, host, port, "client")
        self.chann_ = self.sess_.addChannel()
        self.executor_ = ThreadPoolExecutor()
        self.host = host
        self.port = port
        self.dest_role = dest_role

    # Send val and it's tag to server side, server
    # has cached val when the method return.
    def Remote(self, val, tag):
        msg = {"v": pickle.dumps(val), "tag": tag}
        self.chann_.send(msg)
        _ = self.chann_.recv()
        logger.debug(
            "Send value with tag '{}' to {} finish".format(tag, self.dest_role))

    # Send val and it's tag to server side, client begin the send action
    # in a thread when the the method reutrn but not ensure that server
    # has cached this val. Use 'fut.result()' to wait for server to cache it,
    # this makes send value and other action running in the same time.
    def RemoteAsync(self, val, tag):
        def send_fn(channel, msg):
            channel.send(msg)
            _ = channel.recv()

        msg = {"v": val, "tag": tag}
        fut = self.executor_.submit(send_fn, self.chann_, msg)

        return fut


class ServerChannelProxy:
    def __init__(self, port):
        self.ios_ = IOService()
        self.sess_ = Session(self.ios_, "*", port, "server")
        self.chann_ = self.sess_.addChannel()
        self.executor_ = ThreadPoolExecutor()
        self.recv_cache_ = {}
        self.stop_signal_ = False
        self.recv_loop_fut_ = None

    # Start a recv thread to receive value and cache it.
    def StartRecvLoop(self):
        def recv_loop():
            logger.info("Start recv loop.")
            while (not self.stop_signal_):
                try:
                    msg = self.chann_.recv(block=False)
                except Exception as e:
                    logger.error(e)
                    break

                if msg is None:
                    continue

                key = msg["tag"]
                value = msg["v"]
                if self.recv_cache_.get(key, None) is not None:
                    logger.warn(
                        "Hash entry for tag '{}' is not empty, replace old value".format(key))
                    del self.recv_cache_[key]

                logger.debug("Recv msg with tag '{}'.".format(key))
                self.recv_cache_[key] = value
                self.chann_.send("ok")
            logger.info("Recv loop stops.")

        self.recv_loop_fut_ = self.executor_.submit(recv_loop)

    # Stop recv thread.
    def StopRecvLoop(self):
        self.stop_signal_ = True
        self.recv_loop_fut_.result()
        logger.info("Recv loop already exit, clean cached value.")
        key_list = list(self.recv_cache_.keys())
        for key in key_list:
            del self.recv_cache_[key]
            logger.warn(
                "Remove value with tag '{}', not used until now.".format(key))
        # del self.recv_cache_
        logger.info("Release system resource!")
        self.chann_.socket.close()
        # self.chann_.term()
        # self.chann_.context.destroy()
        # self.chann_.socket.destroy()

    # Get value from cache, and the check will repeat at most 'retries' times,
    # and sleep 0.3s after each check to avoid check all the time.
    def Get(self, tag, max_time=10000, interval=0.1):
        start = time.time()
        while True:
            val = self.recv_cache_.get(tag, None)
            end = time.time()
            if val is not None:
                del self.recv_cache_[tag]
                logger.debug("Get val with tag '{}' finish.".format(tag))
                return pickle.loads(val)

            if (end-start) > max_time:
                logger.warn(
                    "Can't get value for tag '{}', timeout.".format(tag))
                break

            time.sleep(interval)

        return None


class XGB_GUEST:
    def __init__(self, proxy_server=None,
                 proxy_client_host=None,
                 base_score=0.5,
                 max_depth=3,
                 n_estimators=10,
                 learning_rate=0.1,
                 reg_lambda=1,
                 gamma=0,
                 min_child_sample=100,
                 min_child_weight=1,
                 objective='linear',
                 #  channel=None,
                 sid=0,
                 record=0,
                 lookup_table=pd.DataFrame(
                     columns=['record_id', 'feature_id', 'threshold_value'])
                 ):

        # self.channel = channel
        self.proxy_server = proxy_server
        self.proxy_client_host = proxy_client_host

        self.base_score = base_score
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_sample = min_child_sample
        self.min_child_weight = min_child_weight
        self.objective = objective
        self.sid = sid
        self.record = record
        self.lookup_table = lookup_table
        self.lookup_table_sum = {}

    def get_GH(self, X):
        # Calculate G_left、G_right、H_left、H_right under feature segmentation
        GH = pd.DataFrame(
            columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
        i = 0
        for item in [x for x in X.columns if x not in ['g', 'h']]:
            # Categorical variables using greedy algorithm
            # if len(list(set(X[item]))) < 5:
            for cuts in list(set(X[item])):
                if self.min_child_sample:
                    if (X.loc[X[item] < cuts].shape[0] < self.min_child_sample) \
                            | (X.loc[X[item] >= cuts].shape[0] < self.min_child_sample):
                        continue
                GH.loc[i, 'G_left'] = X.loc[X[item] < cuts, 'g'].sum()
                GH.loc[i, 'G_right'] = X.loc[X[item] >= cuts, 'g'].sum()
                GH.loc[i, 'H_left'] = X.loc[X[item] < cuts, 'h'].sum()
                GH.loc[i, 'H_right'] = X.loc[X[item] >= cuts, 'h'].sum()
                GH.loc[i, 'var'] = item
                GH.loc[i, 'cut'] = cuts
                i = i + 1
            # Continuous variables using approximation algorithm
            # else:
            #     old_list = list(set(X[item]))
            #     new_list = []
            #     # four candidate points
            #     j = int(len(old_list) / 4)
            #     for z in range(0, len(old_list), j):
            #         new_list.append(old_list[z])
            #     for cuts in new_list:
            #         if self.min_child_sample:
            #             if (X.loc[X[item] < cuts].shape[0] < self.min_child_sample) \
            #                     | (X.loc[X[item] >= cuts].shape[0] < self.min_child_sample):
            #                 continue
            #         GH.loc[i, 'G_left'] = X.loc[X[item] < cuts, 'g'].sum()
            #         GH.loc[i, 'G_right'] = X.loc[X[item] >= cuts, 'g'].sum()
            #         GH.loc[i, 'H_left'] = X.loc[X[item] < cuts, 'h'].sum()
            #         GH.loc[i, 'H_right'] = X.loc[X[item] >= cuts, 'h'].sum()
            #         GH.loc[i, 'var'] = item
            #         GH.loc[i, 'cut'] = cuts
            #         i = i + 1
        return GH

    def split(self, X, best_var, best_cut, GH_best, w):
        # Calculate the weight of leaf nodes after splitting
        # print("=====guest index====", X.index, best_cut)
        id_left = X.loc[X[best_var] < best_cut].index.tolist()
        w_left = -GH_best['G_left_best'] / \
            (GH_best['H_left_best'] + self.reg_lambda)
        id_right = X.loc[X[best_var] >= best_cut].index.tolist()
        w_right = -GH_best['G_right_best'] / \
            (GH_best['H_right_best'] + self.reg_lambda)
        w[id_left] = w_left
        w[id_right] = w_right
        return w, id_right, id_left, w_right, w_left

    def cart_tree(self, X_guest_gh, mdep):
        print("guest dept", mdep)
        if mdep > self.max_depth:
            return
        # best_var = self.channel.recv()
        best_var = self.proxy_server.Get('best_var')
        if best_var == "True":
            # self.channel.send("True")
            return None

        # self.channel.send("true")
        if best_var in [x for x in X_guest_gh.columns]:
            # var_cut_GH = self.channel.recv()
            var_cut_GH = self.proxy_server.Get('var_cut_GH')
            best_var = var_cut_GH['best_var']
            best_cut = var_cut_GH['best_cut']
            GH_best = var_cut_GH['GH_best']
            f_t = var_cut_GH['f_t']
            self.lookup_table.loc[self.record, 'record_id'] = self.record
            self.lookup_table.loc[self.record, 'feature_id'] = best_var
            self.lookup_table.loc[self.record, 'threshold_value'] = best_cut
            f_t, id_right, id_left, w_right, w_left = self.split(
                X_guest_gh, best_var, best_cut, GH_best, f_t)
            # .reset_index(drop='True'))
            gh_sum_left = self.get_GH(X_guest_gh.loc[id_left])
            # .reset_index(drop='True'))
            gh_sum_right = self.get_GH(X_guest_gh.loc[id_right])

            id_w_gh = {'f_t': f_t, 'id_right': id_right, 'id_left': id_left, 'w_right': w_right,
                       'w_left': w_left, 'gh_sum_right': gh_sum_right, 'gh_sum_left': gh_sum_left,
                       'record_id': self.record, 'party_id': self.sid}
            # self.channel.send(data)
            self.proxy_client_host.Remote(id_w_gh, 'id_w_gh')
            self.record = self.record + 1
            # print("data", type(data), data)
            # time.sleep(5)

        else:
            # id_dic = self.channel.recv()
            id_dic = self.proxy_server.Get('id_dic')
            id_right = id_dic['id_right']

            id_left = id_dic['id_left']
            print("++++++", mdep)
            print("=======guest index-2 ======",
                  len(X_guest_gh.index.tolist()), X_guest_gh.index)

            # .reset_index(drop='True'))
            gh_sum_left = self.get_GH(X_guest_gh.loc[id_left])
            # .reset_index(drop='True'))
            gh_sum_right = self.get_GH(X_guest_gh.loc[id_right])

            gh_sum_dic = {'gh_sum_right': gh_sum_right,
                          'gh_sum_left': gh_sum_left}
            # self.channel.send(
            # )
            self.proxy_client_host.Remote(gh_sum_dic, 'gh_sum_dic')

        # left tree
        print("=====guest shape========",
              X_guest_gh.loc[id_left].shape, X_guest_gh.loc[id_right].shape)
        self.cart_tree(X_guest_gh.loc[id_left], mdep+1)

        # right tree
        self.cart_tree(X_guest_gh.loc[id_right], mdep+1)
        # self.proxy_server.StopRecvLoop()

    def host_record(self, record_id, id_list, tree, X):
        id_after_record = {"id_left": [], "id_right": []}
        record_tree = self.lookup_table_sum[tree + 1]
        feature = record_tree.loc[record_id, "feature_id"]
        threshold_value = record_tree.loc[record_id, 'threshold_value']
        for i in id_list:
            feature_value = X.loc[i, feature]
            if feature_value >= threshold_value:
                id_after_record["id_right"].append(i)
            else:
                id_after_record["id_left"].append(i)

        return id_after_record

    def predict(self, X):
        flag = True
        while (flag):
            # need_record = self.channel.recv()
            need_record = self.proxy_server.Get('need_record')

            # self.channel.send(id_after_record)

            if need_record == -1:
                flag = False
                # self.channel.send(b"finished predict once")
            else:
                record_id = need_record["record_id"]
                id_list = need_record["id"]
                tree = need_record["tree"]
                id_after_record = self.host_record(record_id, id_list, tree, X)
                self.proxy_client_host.Remote(
                    id_after_record, "id_after_record")


class XGB_GUEST_EN:
    def __init__(self, proxy_server=None,
                 proxy_client_host=None,
                 base_score=0.5,
                 max_depth=3,
                 n_estimators=10,
                 learning_rate=0.1,
                 reg_lambda=1,
                 gamma=0,
                 min_child_sample=100,
                 min_child_weight=1,
                 objective='linear',
                 #  channel=None,
                 sid=0,
                 record=0,
                 lookup_table=pd.DataFrame(
                     columns=['record_id', 'feature_id', 'threshold_value'])
                 ):
        # self.channel = channel
        self.proxy_server = proxy_server
        self.proxy_client_host = proxy_client_host

        self.base_score = base_score
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_sample = min_child_sample
        self.min_child_weight = min_child_weight
        self.objective = objective
        self.sid = sid
        self.record = record
        self.lookup_table = lookup_table
        self.lookup_table_sum = {}

    def get_GH(self, X, pub):
        # global
        # def opt_pai_add(x, y):
        #     return opt_paillier_add(pub, x, y)

        # Calculate G_left、G_right、H_left、H_right under feature segmentation
        # arr = np.zeros((X.shape[0] * 10, 6))
        # GH = pd.DataFrame(
        #     arr, columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
        # GH.apply(opt_paillier_encrypt, args=(pub,))
        G_lefts = []
        G_rights = []
        H_lefts = []
        H_rights = []
        vars = []
        cuts = []

        i = 0
        for item in [x for x in X.columns if x not in ['g', 'h']]:
            # Categorical variables using greedy algorithm
            # if len(list(set(X[item]))) < 5:

            for tmp_cut in list(set(X[item])):

                flag = (X[item] < tmp_cut)
                less_sum = sum(flag.astype('int'))
                great_sum = sum((1-flag).astype('int'))

                if self.min_child_sample:
                    if (less_sum < self.min_child_sample) \
                            | (great_sum < self.min_child_sample):
                        continue

                G_left_g = X.loc[flag, 'g'].values.tolist()
                G_right_g = X.loc[(1-flag).astype('bool'), 'g'].values.tolist()
                H_left_h = X.loc[flag, 'h'].values.tolist()
                H_right_h = X.loc[(1-flag).astype('bool'), 'h'].values.tolist()
                print("++++++++++", len(G_left_g), len(G_right_g),
                      len(H_left_h), len(H_right_h))

                # tmp_g_left = functools.reduce(opt_pai_add, G_left_g)
                # opt_paillier_add(pub, x, y)
                tmp_g_left = functools.reduce(
                    lambda x, y: opt_paillier_add(pub, x, y), G_left_g)
                tmp_g_right = functools.reduce(
                    lambda x, y: opt_paillier_add(pub, x, y), G_right_g)
                tmp_h_left = functools.reduce(
                    lambda x, y: opt_paillier_add(pub, x, y), H_left_h)
                tmp_h_right = functools.reduce(
                    lambda x, y: opt_paillier_add(pub, x, y), H_right_h)

                # tmp_g_right = functools.reduce(opt_pai_add, G_right_g)
                # tmp_h_left = functools.reduce(opt_pai_add, H_left_h)
                # tmp_h_right = functools.reduce(opt_pai_add, H_right_h)

                G_lefts.append(tmp_g_left)
                G_rights.append(tmp_g_right)
                H_lefts.append(tmp_h_left)
                H_rights.append(tmp_h_right)
                vars.append(item)
                cuts.append(tmp_cut)

        # GH = pd.DataFrame(
        #     arr, columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
        GH = pd.DataFrame(
            {'G_left': G_lefts, 'G_right': G_rights, 'H_left': H_lefts, 'H_right': H_rights, 'var': vars, 'cut': cuts})

        return GH

        # tmp_g_right = GH.loc[i, 'G_left']

        # if self.min_child_sample:
        #     # if (X.loc[X[item] < cuts].shape[0] < self.min_child_sample) \
        #     #         | (X.loc[X[item] >= cuts].shape[0] < self.min_child_sample):
        #     if (less_sum < self.min_child_sample) \
        #             | (great_sum < self.min_child_sample):
        #         continue
        # for ind in X.index:
        #     if X.loc[ind, item] < cuts:
        #         if GH.loc[i, 'G_left'] == 0:
        #             GH.loc[i, 'G_left'] = X.loc[ind, 'g']
        #         else:
        #             GH.loc[i, 'G_left'] = opt_paillier_add(
        #                 pub, GH.loc[i, 'G_left'], X.loc[ind, 'g'])
        #         if GH.loc[i, 'H_left'] == 0:
        #             GH.loc[i, 'H_left'] = X.loc[ind, 'h']
        #         else:
        #             GH.loc[i, 'H_left'] = opt_paillier_add(
        #                 pub, GH.loc[i, 'H_left'], X.loc[ind, 'h'])
        #     else:
        #         if GH.loc[i, 'G_right'] == 0:
        #             GH.loc[i, 'G_right'] = X.loc[ind, 'g']
        #         else:
        #             GH.loc[i, 'G_right'] = opt_paillier_add(
        #                 pub, GH.loc[i, 'G_right'], X.loc[ind, 'g'])
        #         if GH.loc[i, 'H_right'] == 0:
        #             GH.loc[i, 'H_right'] = X.loc[ind, 'h']
        #         else:
        #             GH.loc[i, 'H_right'] = opt_paillier_add(
        #                 pub, GH.loc[i, 'H_right'], X.loc[ind, 'h'])
        # GH.loc[i, 'var'] = item
        # GH.loc[i, 'cut'] = cuts
        # i = i + 1
        # Continuous variables using approximation algorithm
        # else:
        #     old_list = list(set(X[item]))
        #     new_list = []
        #     # four candidate points
        #     j = int(len(old_list) / 4)
        #     for z in range(0, len(old_list), j):
        #         new_list.append(old_list[z])
        #     for cuts in new_list:
        #         if self.min_child_sample:
        #             if (X.loc[X[item] < cuts].shape[0] < self.min_child_sample) \
        #                     | (X.loc[X[item] >= cuts].shape[0] < self.min_child_sample):
        #                 continue
        #         GH.loc[i, 'G_left'] = X.loc[X[item] < cuts, 'g'].sum()
        #         GH.loc[i, 'G_right'] = X.loc[X[item] >= cuts, 'g'].sum()
        #         GH.loc[i, 'H_left'] = X.loc[X[item] < cuts, 'h'].sum()
        #         GH.loc[i, 'H_right'] = X.loc[X[item] >= cuts, 'h'].sum()
        #         GH.loc[i, 'var'] = item
        #         GH.loc[i, 'cut'] = cuts
        #         i = i + 1
        # return GH

    # def find_split(self, GH):
    #     # Find the feature corresponding to the best split and the split value
    #     GH_best = pd.DataFrame(columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
    #     max_gain = 0
    #     for item in GH.index:
    #         gain = GH.loc[item, 'G_left'] ** 2 / (GH.loc[item, 'H_left'] + self.reg_lambda) + \
    #                GH.loc[item, 'G_right'] ** 2 / (GH.loc[item, 'H_right'] + self.reg_lambda) - \
    #                (GH.loc[item, 'G_left'] + GH.loc[item, 'G_right']) ** 2 / (
    #                        GH.loc[item, 'H_left'] + GH.loc[item, 'H_right'] + + self.reg_lambda)
    #         gain = gain / 2 - self.gamma
    #         if gain > max_gain:
    #             max_gain = gain
    #             GH_best.loc[0, 'G_left'] = GH.loc[item, 'G_left']
    #             GH_best.loc[0, 'G_right'] = GH.loc[item, 'G_right']
    #             GH_best.loc[0, 'H_left'] = GH.loc[item, 'H_left']
    #             GH_best.loc[0, 'H_right'] = GH.loc[item, 'H_right']
    #             GH_best.loc[0, 'var'] = GH.loc[item, 'var']
    #             GH_best.loc[0, 'cut'] = GH.loc[item, 'cut']
    #     return GH_best

    def split(self, X, best_var, best_cut, GH_best, w):
        # Calculate the weight of leaf nodes after splitting
        # print("=====guest index====", X.index, best_cut)
        id_left = X.loc[X[best_var] < best_cut].index.tolist()
        w_left = -GH_best['G_left_best'] / \
            (GH_best['H_left_best'] + self.reg_lambda)
        id_right = X.loc[X[best_var] >= best_cut].index.tolist()
        w_right = -GH_best['G_right_best'] / \
            (GH_best['H_right_best'] + self.reg_lambda)
        w[id_left] = w_left
        w[id_right] = w_right
        return w, id_right, id_left, w_right, w_left

    def cart_tree(self, X_guest_gh, mdep, pub):
        print("guest dept", mdep)
        if mdep > self.max_depth:
            return
        # best_var = self.channel.recv()
        best_var = self.proxy_server.Get('best_var')
        if best_var == "True":
            # self.channel.send("True")
            return None

        # self.channel.send("true")
        if best_var in [x for x in X_guest_gh.columns]:
            # var_cut_GH = self.channel.recv()
            var_cut_GH = self.proxy_server.Get('var_cut_GH')
            best_var = var_cut_GH['best_var']
            best_cut = var_cut_GH['best_cut']
            GH_best = var_cut_GH['GH_best']
            f_t = var_cut_GH['f_t']
            self.lookup_table.loc[self.record, 'record_id'] = self.record
            self.lookup_table.loc[self.record, 'feature_id'] = best_var
            self.lookup_table.loc[self.record, 'threshold_value'] = best_cut
            f_t, id_right, id_left, w_right, w_left = self.split(
                X_guest_gh, best_var, best_cut, GH_best, f_t)

            gh_sum_left = self.get_GH(X_guest_gh.loc[id_left], pub)
            gh_sum_right = self.get_GH(X_guest_gh.loc[id_right], pub)

            id_w_gh = {'f_t': f_t, 'id_right': id_right, 'id_left': id_left, 'w_right': w_right,
                       'w_left': w_left, 'gh_sum_right': gh_sum_right, 'gh_sum_left': gh_sum_left,
                       'record_id': self.record, 'party_id': self.sid}
            # self.channel.send(data)
            self.proxy_client_host.Remote(id_w_gh, 'id_w_gh')

            self.record = self.record + 1
            # print("data", type(data), data)
            # time.sleep(5)

        else:
            # id_dic = self.channel.recv()
            id_dic = self.proxy_server.Get('id_dic')
            id_right = id_dic['id_right']

            id_left = id_dic['id_left']
            print("++++++", mdep)
            print("=======guest index-2 ======",
                  len(X_guest_gh.index.tolist()), X_guest_gh.index)

            # .reset_index(drop='True'))
            gh_sum_left = self.get_GH(X_guest_gh.loc[id_left], pub)
            # .reset_index(drop='True'))
            gh_sum_right = self.get_GH(X_guest_gh.loc[id_right], pub)
            gh_sum_dic = {'gh_sum_right': gh_sum_right,
                          'gh_sum_left': gh_sum_left}
            # self.channel.send(
            # )

        # left tree
        print("=====guest shape========",
              X_guest_gh.loc[id_left].shape, X_guest_gh.loc[id_right].shape)
        self.cart_tree(X_guest_gh.loc[id_left], mdep + 1, pub)

        # right tree
        self.cart_tree(X_guest_gh.loc[id_right], mdep + 1, pub)
        # self.proxy_server.StopRecvLoop()

    def host_record(self, record_id, id_list, tree, X):
        id_after_record = {"id_left": [], "id_right": []}
        record_tree = self.lookup_table_sum[tree + 1]
        feature = record_tree.loc[record_id, "feature_id"]
        threshold_value = record_tree.loc[record_id, 'threshold_value']
        for i in id_list:
            feature_value = X.loc[i, feature]
            if feature_value >= threshold_value:
                id_after_record["id_right"].append(i)
            else:
                id_after_record["id_left"].append(i)

        return id_after_record

    def predict(self, X):
        flag = True
        while (flag):
            # need_record = self.channel.recv()
            need_record = self.proxy_server.Get('need_record')

            if need_record == -1:
                flag = False
                # self.channel.send(b"finished predict once")
            else:
                record_id = need_record["record_id"]
                id_list = need_record["id"]
                tree = need_record["tree"]
                id_after_record = self.host_record(record_id, id_list, tree, X)

                # self.channel.send(id_after_record)
                self.proxy_client_host.Remote(
                    id_after_record, 'id_after_record')


class XGB_HOST:
    def __init__(self, proxy_server=None, proxy_client_guest=None,
                 base_score=0.5,
                 max_depth=3,
                 n_estimators=10,
                 learning_rate=0.1,
                 reg_lambda=1,
                 gamma=0,
                 min_child_sample=100,
                 min_child_weight=1,
                 objective='linear',
                 #  channel=None,
                 sid=0,
                 record=0,
                 lookup_table=pd.DataFrame(
                     columns=['record_id', 'feature_id', 'threshold_value'])
                 ):

        # self.channel = channel
        self.proxy_server = proxy_server
        self.proxy_client_guest = proxy_client_guest
        self.base_score = base_score
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_sample = min_child_sample
        self.min_child_weight = min_child_weight
        self.objective = objective
        self.sid = sid
        self.record = record
        self.lookup_table = lookup_table
        self.tree_structure = {}
        self.lookup_table_sum = {}

    def _grad(self, y_hat, Y):

        if self.objective == 'logistic':
            y_hat = 1.0 / (1.0 + np.exp(-y_hat))
            return y_hat - Y
        elif self.objective == 'linear':
            return y_hat - Y
        else:
            raise KeyError('objective must be linear or logistic!')

    def _hess(self, y_hat, Y):

        if self.objective == 'logistic':
            y_hat = 1.0 / (1.0 + np.exp(-y_hat))
            return y_hat * (1.0 - y_hat)
        elif self.objective == 'linear':
            return np.array([1] * Y.shape[0])
        else:
            raise KeyError('objective must be linear or logistic!')

    def get_gh(self, y_hat, Y):
        # Calculate the g and h of each sample based on the labels of the local data
        gh = pd.DataFrame(columns=['g', 'h'])
        for i in range(0, Y.shape[0]):
            gh['g'] = self._grad(y_hat, Y)
            gh['h'] = self._hess(y_hat, Y)
        return gh

    def get_GH(self, X):
        # Calculate G_left、G_right、H_left、H_right under feature segmentation
        GH = pd.DataFrame(
            columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
        i = 0
        for item in [x for x in X.columns if x not in ['g', 'h', 'y']]:
            # Categorical variables using greedy algorithm
            # if len(list(set(X[item]))) < 5:
            for cuts in list(set(X[item])):
                if self.min_child_sample:
                    if (X.loc[X[item] < cuts].shape[0] < self.min_child_sample) \
                            | (X.loc[X[item] >= cuts].shape[0] < self.min_child_sample):
                        continue
                GH.loc[i, 'G_left'] = X.loc[X[item] < cuts, 'g'].sum()
                GH.loc[i, 'G_right'] = X.loc[X[item] >= cuts, 'g'].sum()
                GH.loc[i, 'H_left'] = X.loc[X[item] < cuts, 'h'].sum()
                GH.loc[i, 'H_right'] = X.loc[X[item] >= cuts, 'h'].sum()
                GH.loc[i, 'var'] = item
                GH.loc[i, 'cut'] = cuts
                i = i + 1
            # Continuous variables using approximation algorithm
            # else:
            #     old_list = list(set(X[item]))
            #     new_list = []
            #     # four candidate points
            #     j = int(len(old_list) / 4)
            #     for z in range(0, len(old_list), j):
            #         new_list.append(old_list[z])
            #     for cuts in new_list:
            #         if self.min_child_sample:
            #             if (X.loc[X[item] < cuts].shape[0] < self.min_child_sample) \
            #                     | (X.loc[X[item] >= cuts].shape[0] < self.min_child_sample):
            #                 continue
            #         GH.loc[i, 'G_left'] = X.loc[X[item] < cuts, 'g'].sum()
            #         GH.loc[i, 'G_right'] = X.loc[X[item] >= cuts, 'g'].sum()
            #         GH.loc[i, 'H_left'] = X.loc[X[item] < cuts, 'h'].sum()
            #         GH.loc[i, 'H_right'] = X.loc[X[item] >= cuts, 'h'].sum()
            #         GH.loc[i, 'var'] = item
            #         GH.loc[i, 'cut'] = cuts
            #         i = i + 1
        return GH

    def find_split(self, GH_host, GH_guest):
        # Find the feature corresponding to the best split and the split value
        best_var, best_cut = None, None
        GH_best = {}
        max_gain = 0
        GH = pd.concat([GH_host, GH_guest], axis=0, ignore_index=True)
        for item in GH.index:
            gain = GH.loc[item, 'G_left'] ** 2 / (GH.loc[item, 'H_left'] + self.reg_lambda) + \
                GH.loc[item, 'G_right'] ** 2 / (GH.loc[item, 'H_right'] + self.reg_lambda) - \
                (GH.loc[item, 'G_left'] + GH.loc[item, 'G_right']) ** 2 / (
                GH.loc[item, 'H_left'] + GH.loc[item, 'H_right'] + + self.reg_lambda)
            gain = gain / 2 - self.gamma
            if gain > max_gain:
                best_var = GH.loc[item, 'var']
                best_cut = GH.loc[item, 'cut']
                max_gain = gain
                GH_best['G_left_best'] = GH.loc[item, 'G_left']
                GH_best['G_right_best'] = GH.loc[item, 'G_right']
                GH_best['H_left_best'] = GH.loc[item, 'H_left']
                GH_best['H_right_best'] = GH.loc[item, 'H_right']
        return best_var, best_cut, GH_best

    def split(self, X, best_var, best_cut, GH_best, w):
        # Calculate the weight of leaf nodes after splitting
        # print("++++++host index-1+++++++", len(X.index.tolist()), X.index)
        id_left = X.loc[X[best_var] < best_cut].index.tolist()
        w_left = -GH_best['G_left_best'] / \
            (GH_best['H_left_best'] + self.reg_lambda)
        id_right = X.loc[X[best_var] >= best_cut].index.tolist()
        w_right = -GH_best['G_right_best'] / \
            (GH_best['H_right_best'] + self.reg_lambda)
        w[id_left] = w_left
        w[id_right] = w_right
        return w, id_right, id_left, w_right, w_left

    def xgb_tree(self, X_host, GH_guest, gh, f_t, m_dpth):
        print("=====host mdep===", m_dpth)
        if m_dpth > self.max_depth:
            return
        X_host = X_host
        gh = gh
        X_host_gh = pd.concat([X_host, gh], axis=1)
        GH_host = self.get_GH(X_host_gh)

        best_var, best_cut, GH_best = self.find_split(GH_host, GH_guest)
        if best_var is None:
            # self.proxy_client_guest.Remote("True", 'best_var')
            best_var = "True"

        self.proxy_client_guest.Remote(best_var, 'best_var')
        print("best_var: ", best_var)
        if best_var == "True":
            # self.proxy_server.StopRecvLoop()
            return None
        else:
            # self.channel.send(best_var)
            # self.proxy_client_guest.Remote(best_var, 'best_var')
            # # flag = self.channel.recv()
            # if flag:
            if best_var not in [x for x in X_host.columns]:
                var_cut_GH = {'best_var': best_var, 'best_cut': best_cut,
                              'GH_best': GH_best, 'f_t': f_t}
                # self.channel.send(data)
                self.proxy_client_guest.Remote(var_cut_GH, 'var_cut_GH')

                # id_w_gh = self.channel.recv()
                id_w_gh = self.proxy_server.Get('id_w_gh')
                f_t = id_w_gh['f_t']
                id_right = id_w_gh['id_right']
                id_left = id_w_gh['id_left']
                w_right = id_w_gh['w_right']
                w_left = id_w_gh['w_left']
                record_id = id_w_gh['record_id']
                party_id = id_w_gh['party_id']
                tree_structure = {(party_id, record_id): {}}
                gh_sum_right = id_w_gh['gh_sum_right']
                gh_sum_left = id_w_gh['gh_sum_left']

            else:
                self.lookup_table.loc[self.record, 'record_id'] = self.record
                self.lookup_table.loc[self.record, 'feature_id'] = best_var
                self.lookup_table.loc[self.record,
                                      'threshold_value'] = best_cut
                record_id = self.record
                party_id = self.sid
                tree_structure = {(party_id, record_id): {}}
                self.record = self.record + 1
                f_t, id_right, id_left, w_right, w_left = self.split(
                    X_host, best_var, best_cut, GH_best, f_t)
                print("host split", X_host.index)
                id_dic = {'id_right': id_right,
                          'id_left': id_left, "best_cut": best_cut}
                # self.channel.send(
                # )
                self.proxy_client_guest.Remote(id_dic, 'id_dic')
                # gh_sum_dic = self.channel.recv()
                gh_sum_dic = self.proxy_server.Get('gh_sum_dic')
                gh_sum_left = gh_sum_dic['gh_sum_left']
                gh_sum_right = gh_sum_dic['gh_sum_right']

            print("=====x host index=====", X_host.index)
            print("host shape",
                  X_host.loc[id_left].shape, X_host.loc[id_right].shape)

            result_left = self.xgb_tree(X_host.loc[id_left],
                                        gh_sum_left,
                                        gh.loc[id_left],
                                        f_t,
                                        m_dpth + 1)
            if isinstance(result_left, tuple):
                tree_structure[(party_id, record_id)][(
                    'left', w_left)] = copy.deepcopy(result_left[0])
                f_t = result_left[1]
            else:
                tree_structure[(party_id, record_id)][(
                    'left', w_left)] = copy.deepcopy(result_left)
            result_right = self.xgb_tree(X_host.loc[id_right],
                                         gh_sum_right,
                                         gh.loc[id_right],
                                         f_t,
                                         m_dpth + 1)
            if isinstance(result_right, tuple):
                tree_structure[(party_id, record_id)][(
                    'right', w_right)] = copy.deepcopy(result_right[0])
                f_t = result_right[1]
            else:
                tree_structure[(party_id, record_id)][(
                    'right', w_right)] = copy.deepcopy(result_right)
        # self.proxy_server.StopRecvLoop()

        return tree_structure, f_t

    def _get_tree_node_w(self, X, tree, lookup_table, w, t):
        if not tree is None:
            if isinstance(tree, tuple):
                tree = tree[0]
            k = list(tree.keys())[0]
            party_id, record_id = k[0], k[1]
            id = X.index.tolist()
            if party_id != self.sid:
                # self.channel.send(
                #     )
                need_record = {'id': id, 'record_id': record_id, 'tree': t}
                self.proxy_client_guest.Remote(need_record, 'need_record')
                # split_id = self.channel.recv()
                split_id = self.proxy_server.Get('id_after_record')
                id_left = split_id['id_left']
                id_right = split_id['id_right']
                if id_left == [] or id_right == []:
                    return
                else:
                    X_left = X.loc[id_left, :]
                    X_right = X.loc[id_right, :]
                for kk in tree[k].keys():
                    if kk[0] == 'left':
                        tree_left = tree[k][kk]
                        w[id_left] = kk[1]
                    elif kk[0] == 'right':
                        tree_right = tree[k][kk]
                        w[id_right] = kk[1]

                self._get_tree_node_w(X_left, tree_left, lookup_table, w, t)
                self._get_tree_node_w(X_right, tree_right, lookup_table, w, t)
            else:
                for index in lookup_table.index:
                    if lookup_table.loc[index, 'record_id'] == record_id:
                        var = lookup_table.loc[index, 'feature_id']
                        cut = lookup_table.loc[index, 'threshold_value']
                        X_left = X.loc[X[var] < cut]
                        id_left = X_left.index.tolist()
                        X_right = X.loc[X[var] >= cut]
                        id_right = X_right.index.tolist()
                        if id_left == [] or id_right == []:
                            return
                        for kk in tree[k].keys():
                            if kk[0] == 'left':
                                tree_left = tree[k][kk]
                                w[id_left] = kk[1]
                            elif kk[0] == 'right':
                                tree_right = tree[k][kk]
                                w[id_right] = kk[1]

                        self._get_tree_node_w(
                            X_left, tree_left, lookup_table, w, t)
                        self._get_tree_node_w(
                            X_right, tree_right, lookup_table, w, t)

    def predict_raw(self, X: pd.DataFrame):
        X = X.reset_index(drop='True')
        Y = pd.Series([self.base_score] * X.shape[0])

        for t in range(self.n_estimators):
            tree = self.tree_structure[t + 1]
            lookup_table = self.lookup_table_sum[t + 1]
            y_t = pd.Series([0] * X.shape[0])
            self._get_tree_node_w(X, tree, lookup_table, y_t, t)
            Y = Y + self.learning_rate * y_t

        # self.channel.send(-1)
        self.proxy_client_guest.Remote(-1, 'need_record')
        # print(self.channel.recv())
        return Y

    def predict_prob(self, X: pd.DataFrame):

        Y = self.predict_raw(X)

        def sigmoid(x): return 1 / (1 + np.exp(-x))

        Y = Y.apply(sigmoid)

        return Y


class XGB_HOST_EN:
    def __init__(self, proxy_server=None, proxy_client_guest=None,
                 base_score=0.5,
                 max_depth=3,
                 n_estimators=10,
                 learning_rate=0.1,
                 reg_lambda=1,
                 gamma=0,
                 min_child_sample=100,
                 min_child_weight=1,
                 objective='linear',
                 #  channel=None,
                 random_seed=112,
                 sid=0,
                 record=0,
                 lookup_table=pd.DataFrame(
                     columns=['record_id', 'feature_id', 'threshold_value'])
                 ):

        # self.channel = channel
        self.proxy_server = proxy_server
        self.proxy_client_guest = proxy_client_guest
        self.base_score = base_score
        self.max_depth = max_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.gamma = gamma
        self.min_child_sample = min_child_sample
        self.min_child_weight = min_child_weight
        self.objective = objective
        pub, prv = opt_paillier_keygen(random_seed)
        self.pub = pub
        self.prv = prv
        self.sid = sid
        self.record = record
        self.lookup_table = lookup_table
        self.tree_structure = {}
        self.lookup_table_sum = {}

    def _grad(self, y_hat, Y):

        if self.objective == 'logistic':
            y_hat = 1.0 / (1.0 + np.exp(-y_hat))
            return y_hat - Y
        elif self.objective == 'linear':
            return (y_hat - Y) * 10000
        else:
            raise KeyError('objective must be linear or logistic!')

    def _hess(self, y_hat, Y):

        if self.objective == 'logistic':
            y_hat = 1.0 / (1.0 + np.exp(-y_hat))
            return y_hat * (1.0 - y_hat)
        elif self.objective == 'linear':
            return np.array([10000] * Y.shape[0])
        else:
            raise KeyError('objective must be linear or logistic!')

    def get_gh(self, y_hat, Y):
        # Calculate the g and h of each sample based on the labels of the local data
        gh = pd.DataFrame(columns=['g', 'h'])
        for i in range(0, Y.shape[0]):
            gh['g'] = self._grad(y_hat, Y)
            gh['h'] = self._hess(y_hat, Y)

        return gh

    def get_GH(self, X):
        # Calculate G_left、G_right、H_left、H_right under feature segmentation
        GH = pd.DataFrame(
            columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
        i = 0
        for item in [x for x in X.columns if x not in ['g', 'h', 'y']]:
            # Categorical variables using greedy algorithm
            # if len(list(set(X[item]))) < 5:
            for cuts in list(set(X[item])):
                if self.min_child_sample:
                    if (X.loc[X[item] < cuts].shape[0] < self.min_child_sample) \
                            | (X.loc[X[item] >= cuts].shape[0] < self.min_child_sample):
                        continue
                GH.loc[i, 'G_left'] = X.loc[X[item] < cuts, 'g'].sum()
                GH.loc[i, 'G_right'] = X.loc[X[item] >= cuts, 'g'].sum()
                GH.loc[i, 'H_left'] = X.loc[X[item] < cuts, 'h'].sum()
                GH.loc[i, 'H_right'] = X.loc[X[item] >= cuts, 'h'].sum()
                GH.loc[i, 'var'] = item
                GH.loc[i, 'cut'] = cuts
                i = i + 1
            # Continuous variables using approximation algorithm
            # else:
            #     old_list = list(set(X[item]))
            #     new_list = []
            #     # four candidate points
            #     j = int(len(old_list) / 4)
            #     for z in range(0, len(old_list), j):
            #         new_list.append(old_list[z])
            #     for cuts in new_list:
            #         if self.min_child_sample:
            #             if (X.loc[X[item] < cuts].shape[0] < self.min_child_sample) \
            #                     | (X.loc[X[item] >= cuts].shape[0] < self.min_child_sample):
            #                 continue
            #         GH.loc[i, 'G_left'] = X.loc[X[item] < cuts, 'g'].sum()
            #         GH.loc[i, 'G_right'] = X.loc[X[item] >= cuts, 'g'].sum()
            #         GH.loc[i, 'H_left'] = X.loc[X[item] < cuts, 'h'].sum()
            #         GH.loc[i, 'H_right'] = X.loc[X[item] >= cuts, 'h'].sum()
            #         GH.loc[i, 'var'] = item
            #         GH.loc[i, 'cut'] = cuts
            #         i = i + 1
        return GH

    def find_split(self, GH_host, GH_guest):
        # Find the feature corresponding to the best split and the split value
        best_var, best_cut = None, None
        GH_best = {}
        max_gain = 0
        GH = pd.concat([GH_host, GH_guest], axis=0, ignore_index=True)
        for item in GH.index:
            gain = GH.loc[item, 'G_left'] ** 2 / (GH.loc[item, 'H_left'] + self.reg_lambda) + \
                GH.loc[item, 'G_right'] ** 2 / (GH.loc[item, 'H_right'] + self.reg_lambda) - \
                (GH.loc[item, 'G_left'] + GH.loc[item, 'G_right']) ** 2 / (
                GH.loc[item, 'H_left'] + GH.loc[item, 'H_right'] + + self.reg_lambda)
            gain = gain / 2 - self.gamma
            if gain > max_gain:
                best_var = GH.loc[item, 'var']
                best_cut = GH.loc[item, 'cut']
                max_gain = gain
                GH_best['G_left_best'] = GH.loc[item, 'G_left']
                GH_best['G_right_best'] = GH.loc[item, 'G_right']
                GH_best['H_left_best'] = GH.loc[item, 'H_left']
                GH_best['H_right_best'] = GH.loc[item, 'H_right']
        return best_var, best_cut, GH_best

    def split(self, X, best_var, best_cut, GH_best, w):
        # Calculate the weight of leaf nodes after splitting
        # print("++++++host index-1+++++++", len(X.index.tolist()), X.index)
        id_left = X.loc[X[best_var] < best_cut].index.tolist()
        w_left = -GH_best['G_left_best'] / \
            (GH_best['H_left_best'] + self.reg_lambda)
        id_right = X.loc[X[best_var] >= best_cut].index.tolist()
        w_right = -GH_best['G_right_best'] / \
            (GH_best['H_right_best'] + self.reg_lambda)
        w[id_left] = w_left
        w[id_right] = w_right
        return w, id_right, id_left, w_right, w_left

    def xgb_tree(self, X_host, GH_guest, gh, f_t, m_dpth):
        print("=====host mdep===", m_dpth)
        if m_dpth > self.max_depth:
            return
        X_host = X_host
        gh = gh
        X_host_gh = pd.concat([X_host, gh], axis=1)
        GH_host = self.get_GH(X_host_gh)

        best_var, best_cut, GH_best = self.find_split(GH_host, GH_guest)
        if best_var is None:
            best_var = "True"

        self.proxy_client_guest.Remote(best_var, 'best_var')
        if best_var == "True":
            # self.proxy_server.StopRecvLoop()
            return None
            # self.channel.send("True")
            # self.channel.recv()
            # return None

        # self.channel.send(best_var)
        # flag = self.channel.recv()
        else:
            if best_var not in [x for x in X_host.columns]:
                var_cut_GH = {'best_var': best_var, 'best_cut': best_cut,
                              'GH_best': GH_best, 'f_t': f_t}

                # self.channel.send(data)
                self.proxy_client_guest.Remote(var_cut_GH, "var_cut_GH")
                # id_w_gh = self.channel.recv()
                id_w_gh = self.proxy_server.Get('id_w_gh')
                f_t = id_w_gh['f_t']
                id_right = id_w_gh['id_right']
                id_left = id_w_gh['id_left']
                w_right = id_w_gh['w_right']
                w_left = id_w_gh['w_left']
                record_id = id_w_gh['record_id']
                party_id = id_w_gh['party_id']
                tree_structure = {(party_id, record_id): {}}
                gh_sum_right_en = id_w_gh['gh_sum_right']
                vars = gh_sum_right_en.pop('var')
                cuts = gh_sum_right_en.pop('cut')
                gh_sum_right_en_li = gh_sum_right_en.values.tolist()

                gh_sum_right_dec_li = list(map(
                    lambda x: phe_map_dec(self.pub, self.prv, x), gh_sum_right_en_li))
                gh_sum_right = pd.DataFrame(
                    {'gh_sum_right': gh_sum_right_dec_li})
                # gh_sum_right = gh_sum_right_en.apply(opt_paillier_decrypt_crt,
                #                                      args=(self.pub,
                #                                            self.prv))
                gh_sum_right = pd.concat([gh_sum_right, vars, cuts], axis=1)

                # gh_sum_right = pd.DataFrame(
                #     columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
                # for item in [x for x in gh_sum_right_en.columns if x not in ['cut', 'var']]:
                #     for index in gh_sum_right_en.index:
                #         if gh_sum_right_en.loc[index, item] == 0:
                #             gh_sum_right.loc[index, item] = 0
                #         else:
                #             gh_sum_right.loc[index, item] = opt_paillier_decrypt_crt(self.pub, self.prv,
                #                                                                      gh_sum_right_en.loc[index, item])
                # for item in [x for x in gh_sum_right_en.columns if x not in ['G_left', 'G_right', 'H_left', 'H_right']]:
                #     for index in gh_sum_right_en.index:
                #         gh_sum_right.loc[index,
                #                          item] = gh_sum_right_en.loc[index, item]
                gh_sum_left_en = id_w_gh['gh_sum_left']
                vars1 = gh_sum_left_en['var']
                cuts1 = gh_sum_left_en['cut']

                gh_sum_left_en_li = gh_sum_left_en.values.tolist()

                gh_sum_left_dec_li = list(map(
                    lambda x: phe_map_dec(self.pub, self.prv, x), gh_sum_left_en_li))

                gh_sum_left = pd.DataFrame({'gh_sum_left': gh_sum_left_dec_li})

                # gh_sum_left = gh_sum_left_en.apply(opt_paillier_decrypt_crt,
                #                                    args=(self.pub,
                #                                          self.prv))

                gh_sum_left = pd.concat([gh_sum_left, vars1, cuts1], axis=1)
                # gh_sum_left = pd.DataFrame(
                #     columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
                # for item in [x for x in gh_sum_left_en.columns if x not in ['cut', 'var']]:
                #     for index in gh_sum_left_en.index:
                #         if gh_sum_left_en.loc[index, item] == 0:
                #             gh_sum_left.loc[index, item] = 0
                #         else:
                #             gh_sum_left.loc[index, item] = opt_paillier_decrypt_crt(self.pub, self.prv,
                #                                                                     gh_sum_left_en.loc[index, item])
                # for item in [x for x in gh_sum_left_en.columns if x not in ['G_left', 'G_right', 'H_left', 'H_right']]:
                #     for index in gh_sum_left_en.index:
                #         gh_sum_left.loc[index,
                #                         item] = gh_sum_left_en.loc[index, item]
            else:
                self.lookup_table.loc[self.record, 'record_id'] = self.record
                self.lookup_table.loc[self.record, 'feature_id'] = best_var
                self.lookup_table.loc[self.record,
                                      'threshold_value'] = best_cut
                record_id = self.record
                party_id = self.sid
                tree_structure = {(party_id, record_id): {}}
                self.record = self.record + 1
                f_t, id_right, id_left, w_right, w_left = self.split(
                    X_host, best_var, best_cut, GH_best, f_t)
                print("host split", X_host.index)
                id_dic = {'id_right': id_right,
                          'id_left': id_left, "best_cut": best_cut}
                self.proxy_client_guest.Remote(id_dic, 'id_dic')
                # self.channel.send(
                # )
                # gh_sum_dic = self.channel.recv()
                gh_sum_dic = self.proxy_server.Get('gh_sum_dic')
                gh_sum_right_en = gh_sum_dic['gh_sum_right']

                vars = gh_sum_right_en.pop('var')
                cuts = gh_sum_right_en.pop('cut')
                gh_sum_right_en_li = gh_sum_right_en.values.tolist()

                gh_sum_right_dec_li = list(map(
                    lambda x: phe_map_dec(self.pub, self.prv, x), gh_sum_right_en_li))
                gh_sum_right = pd.DataFrame(
                    {'gh_sum_right': gh_sum_right_dec_li})
                # gh_sum_right = gh_sum_right_en.apply(opt_paillier_decrypt_crt,
                #                                      args=(self.pub,
                #                                            self.prv))
                gh_sum_right = pd.concat([gh_sum_right, vars, cuts], axis=1)

                # gh_sum_right = pd.DataFrame(
                #     columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
                # for item in [x for x in gh_sum_right_en.columns if x not in ['cut', 'var']]:
                #     for index in gh_sum_right_en.index:
                #         if gh_sum_right_en.loc[index, item] == 0:
                #             gh_sum_right.loc[index, item] = 0
                #         else:
                #             gh_sum_right.loc[index, item] = opt_paillier_decrypt_crt(self.pub, self.prv,
                #                                                                      gh_sum_right_en.loc[index, item])
                # for item in [x for x in gh_sum_right_en.columns if x not in ['G_left', 'G_right', 'H_left', 'H_right']]:
                #     for index in gh_sum_right_en.index:
                #         gh_sum_right.loc[index,
                #                          item] = gh_sum_right_en.loc[index, item]
                gh_sum_left_en = gh_sum_dic['gh_sum_left']

                vars1 = gh_sum_left_en['var']
                cuts1 = gh_sum_left_en['cut']
                gh_sum_right_en_li = gh_sum_right_en.values.tolist()

                gh_sum_right_dec_li = list(map(
                    lambda x: phe_map_dec(self.pub, self.prv, x), gh_sum_right_en_li))
                gh_sum_right = pd.DataFrame(
                    {'gh_sum_right': gh_sum_right_dec_li})
                # gh_sum_left = gh_sum_left_en.apply(opt_paillier_decrypt_crt,
                #                                    args=(self.pub,
                #                                          self.prv))

                gh_sum_left = pd.concat([gh_sum_left, vars1, cuts1], axis=1)
                # gh_sum_left = pd.DataFrame(
                #     columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
                # for item in [x for x in gh_sum_left_en.columns if x not in ['cut', 'var']]:
                #     for index in gh_sum_left_en.index:
                #         if gh_sum_left_en.loc[index, item] == 0:
                #             gh_sum_left.loc[index, item] = 0
                #         else:
                #             gh_sum_left.loc[index, item] = opt_paillier_decrypt_crt(self.pub, self.prv,
                #                                                                     gh_sum_left_en.loc[index, item])
                # for item in [x for x in gh_sum_left_en.columns if x not in ['G_left', 'G_right', 'H_left', 'H_right']]:
                #     for index in gh_sum_left_en.index:
                #         gh_sum_left.loc[index,
                #                         item] = gh_sum_left_en.loc[index, item]

            print("=====x host index=====", X_host.index)
            print("host shape",
                  X_host.loc[id_left].shape, X_host.loc[id_right].shape)

            result_left = self.xgb_tree(X_host.loc[id_left],
                                        gh_sum_left,
                                        gh.loc[id_left],
                                        f_t,
                                        m_dpth + 1)
            if isinstance(result_left, tuple):
                tree_structure[(party_id, record_id)][(
                    'left', w_left)] = copy.deepcopy(result_left[0])
                f_t = result_left[1]
            else:
                tree_structure[(party_id, record_id)][(
                    'left', w_left)] = copy.deepcopy(result_left)
            result_right = self.xgb_tree(X_host.loc[id_right],
                                         gh_sum_right,
                                         gh.loc[id_right],
                                         f_t,
                                         m_dpth + 1)
            if isinstance(result_right, tuple):
                tree_structure[(party_id, record_id)][(
                    'right', w_right)] = copy.deepcopy(result_right[0])
                f_t = result_right[1]
            else:
                tree_structure[(party_id, record_id)][(
                    'right', w_right)] = copy.deepcopy(result_right)
        # self.proxy_server.StopRecvLoop()

        return tree_structure, f_t

    def _get_tree_node_w(self, X, tree, lookup_table, w, t):
        if not tree is None:
            if isinstance(tree, tuple):
                tree = tree[0]
            k = list(tree.keys())[0]
            party_id, record_id = k[0], k[1]
            id = X.index.tolist()
            if party_id != self.sid:
                need_record = {'id': id, 'record_id': record_id, 'tree': t}
                self.proxy_client_guest.Remote(need_record, 'need_record')

                # self.channel.send(
                #     )
                # split_id = self.channel.recv()
                split_id = self.proxy_server.Get('id_after_record')
                id_left = split_id['id_left']
                id_right = split_id['id_right']
                if id_left == [] or id_right == []:
                    return
                else:
                    X_left = X.loc[id_left, :]
                    X_right = X.loc[id_right, :]
                for kk in tree[k].keys():
                    if kk[0] == 'left':
                        tree_left = tree[k][kk]
                        w[id_left] = kk[1]
                    elif kk[0] == 'right':
                        tree_right = tree[k][kk]
                        w[id_right] = kk[1]

                self._get_tree_node_w(X_left, tree_left, lookup_table, w, t)
                self._get_tree_node_w(X_right, tree_right, lookup_table, w, t)
            else:
                for index in lookup_table.index:
                    if lookup_table.loc[index, 'record_id'] == record_id:
                        var = lookup_table.loc[index, 'feature_id']
                        cut = lookup_table.loc[index, 'threshold_value']
                        X_left = X.loc[X[var] < cut]
                        id_left = X_left.index.tolist()
                        X_right = X.loc[X[var] >= cut]
                        id_right = X_right.index.tolist()
                        if id_left == [] or id_right == []:
                            return
                        for kk in tree[k].keys():
                            if kk[0] == 'left':
                                tree_left = tree[k][kk]
                                w[id_left] = kk[1]
                            elif kk[0] == 'right':
                                tree_right = tree[k][kk]
                                w[id_right] = kk[1]

                        self._get_tree_node_w(
                            X_left, tree_left, lookup_table, w, t)
                        self._get_tree_node_w(
                            X_right, tree_right, lookup_table, w, t)

    def predict_raw(self, X: pd.DataFrame):
        X = X.reset_index(drop='True')
        Y = pd.Series([self.base_score] * X.shape[0])

        for t in range(self.n_estimators):
            tree = self.tree_structure[t + 1]
            lookup_table = self.lookup_table_sum[t + 1]
            y_t = pd.Series([0] * X.shape[0])
            self._get_tree_node_w(X, tree, lookup_table, y_t, t)
            Y = Y + self.learning_rate * y_t

        # self.channel.send(-1)
        self.proxy_client_guest.Remote(-1, 'need_record')
        # print(self.channel.recv())
        return Y

    def predict_prob(self, X: pd.DataFrame):

        Y = self.predict_raw(X)

        def sigmoid(x): return 1 / (1 + np.exp(-x))

        Y = Y.apply(sigmoid)

        return Y


def get_logger(name):
    LOG_FORMAT = "[%(asctime)s][%(filename)s:%(lineno)d][%(levelname)s] %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(level=logging.DEBUG,
                        format=LOG_FORMAT, datefmt=DATE_FORMAT)
    logger = logging.getLogger(name)
    return logger


logger = get_logger("hetero_xgb")

dataset.define("guest_dataset")
dataset.define("label_dataset")

ph.context.Context.func_params_map = {
    "xgb_host_logic": ("paillier",),
    "xgb_guest_logic": ("paillier",)
}

# Number of tree to fit.
num_tree = 2
# Max depth of each tree.
max_depth = 5


@ ph.context.function(role='host', protocol='xgboost', datasets=['label_dataset'], port='8000', task_type="regression")
def xgb_host_logic(cry_pri="paillier"):
    # def xgb_host_logic(cry_pri="plaintext"):
    start = time.time()
    logger.info("start xgb host logic...")

    role_node_map = ph.context.Context.get_role_node_map()
    node_addr_map = ph.context.Context.get_node_addr_map()
    dataset_map = ph.context.Context.dataset_map

    logger.debug(
        "dataset_map {}".format(dataset_map))

    logger.debug(
        "role_nodeid_map {}".format(role_node_map))

    logger.debug(
        "node_addr_map {}".format(node_addr_map))

    data_key = list(dataset_map.keys())[0]

    eva_type = ph.context.Context.params_map.get("taskType", None)
    if eva_type is None:
        logger.warn(
            "taskType is not specified, set to default value 'regression'.")
        eva_type = "regression"

    eva_type = eva_type.lower()
    if eva_type != "classification" and eva_type != "regression":
        logger.error(
            "Invalid value of taskType, possible value is 'regression', 'classification'.")
        return

    logger.info("Current task type is {}.".format(eva_type))

    # 读取注册数据
    data = ph.dataset.read(dataset_key=data_key).df_data

    # y = data.pop('Class').values

    print("host data: ", data)

    if len(role_node_map["host"]) != 1:
        logger.error("Current node of host party: {}".format(
            role_node_map["host"]))
        logger.error("In hetero XGB, only dataset of host party has label, "
                     "so host party must have one, make sure it.")
        return

    host_nodes = role_node_map["host"]
    host_port = node_addr_map[host_nodes[0]].split(":")[1]

    guest_nodes = role_node_map["guest"]
    guest_ip, guest_port = node_addr_map[guest_nodes[0]].split(":")

    proxy_server = ServerChannelProxy(host_port)
    proxy_server.StartRecvLoop()

    proxy_client_guest = ClientChannelProxy(guest_ip, guest_port,
                                            "guest")

    Y = data.pop('Class').values
    X_host = data.copy()
    X_host.pop('Sample code number')
    # public_k, priv_k = paillier.generate_paillier_keypair()
    # logger.debug("paillier pub key is : {}".format(public_k))
    # print("paillier pub key is :", public_k)

    if cry_pri == "paillier":
        xgb_host = XGB_HOST_EN(n_estimators=num_tree, max_depth=max_depth, reg_lambda=1,
                               sid=0, min_child_weight=1, objective='linear', proxy_server=proxy_server, proxy_client_guest=proxy_client_guest)
        # channel.recv()
        # xgb_host.channel.send(xgb_host.pub)
        proxy_client_guest.Remote(xgb_host.pub, "xgb_pub")
        # proxy_client_guest.Remote(public_k, "xgb_pub")
        # print(xgb_host.channel.recv())
        y_hat = np.array([0.5] * Y.shape[0])

        for t in range(xgb_host.n_estimators):
            print("Begin to trian tree: ", t + 1)

            xgb_host.record = 0
            xgb_host.lookup_table = pd.DataFrame(
                columns=['record_id', 'feature_id', 'threshold_value'])
            f_t = pd.Series([0] * Y.shape[0])
            gh = xgb_host.get_gh(y_hat, Y)

            # convert g and h to ints
            ratio = 10**3
            gh = (gh * ratio).astype('int')
            # gh_en = pd.DataFrame(columns=['g', 'h'])
            flat_gh = gh.values.flatten().tolist()
            enc_flat_gh = list(
                map(lambda x: phe_map_enc(xgb_host.pub, xgb_host.prv, x), flat_gh))
            enc_gh = np.array(enc_flat_gh).reshape((-1, 2))
            enc_gh_df = pd.DataFrame(enc_gh, columns=['g', 'h'])

            # gh.apply(opt_paillier_encrypt_crt,
            #          args=(xgb_host.pub, xgb_host.prv))

            gh_en = enc_gh_df
            # for item in gh.columns:
            #     for index in gh.index:
            #         gh_en.loc[index, item] = opt_paillier_encrypt_crt(xgb_host.pub, xgb_host.prv,
            #                                                           int(gh.loc[index, item]))
            print("Encrypt finish.")
            print("gh_en: ", (gh_en))

            proxy_client_guest.Remote(gh_en, "gh_en")

            # GH_guest_en = xgb_host.channel.recv()
            GH_guest_en = proxy_server.Get('gh_sum')

            vars = GH_guest_en.pop('var')
            cuts = GH_guest_en.pop('cut')
            tmp_shape = GH_guest_en.shape
            tmp_columns = GH_guest_en.columns
            GH_guest_en_li = GH_guest_en.values.flatten().tolist()
            GH_guest_dec_li = list(map(
                lambda x: phe_map_dec(xgb_host.pub, xgb_host.prv, x), GH_guest_en_li))
            # GH_guest = GH_guest_en.apply(
            #     opt_paillier_decrypt_crt, args=(xgb_host.pub, xgb_host.prv))
            GH_guest_dec = np.array(GH_guest_dec_li).reshape(tmp_shape)
            GH_guest = pd.DataFrame(GH_guest_dec, columns=tmp_columns)

            GH_guest = pd.concat([GH_guest, vars, cuts], axis=1)
            print("GH_guest: ", GH_guest)

            # GH_guest = pd.DataFrame(
            #     columns=['G_left', 'G_right', 'H_left', 'H_right', 'var', 'cut'])
            # for item in [x for x in GH_guest_en.columns if x not in ['cut', 'var']]:
            #     for index in GH_guest_en.index:
            #         if GH_guest_en.loc[index, item] == 0:
            #             GH_guest.loc[index, item] = 0
            #         else:
            #             GH_guest.loc[index, item] = opt_paillier_decrypt_crt(xgb_host.pub, xgb_host.prv,
            #                                                                  GH_guest_en.loc[index, item])

            # logger.info("Decrypt finish.")

            # for item in [x for x in GH_guest_en.columns if x not in ['G_left', 'G_right', 'H_left', 'H_right']]:
            #     for index in GH_guest_en.index:
            #         GH_guest.loc[index, item] = GH_guest_en.loc[index, item]

            xgb_host.tree_structure[t + 1], f_t = xgb_host.xgb_tree(X_host, GH_guest, gh, f_t, 0)  # noqa
            xgb_host.lookup_table_sum[t + 1] = xgb_host.lookup_table
            y_hat = y_hat + xgb_host.learning_rate * f_t

            logger.info("Finish to trian tree {}.".format(t + 1))

        predict_file_path = ph.context.Context.get_predict_file_path()
        indicator_file_path = ph.context.Context.get_indicator_file_path()
        model_file_path = ph.context.Context.get_model_file_path()
        lookup_file_path = ph.context.Context.get_host_lookup_file_path()

        with open(model_file_path, 'wb') as fm:
            pickle.dump(xgb_host.tree_structure, fm)
        with open(lookup_file_path, 'wb') as fl:
            pickle.dump(xgb_host.lookup_table_sum, fl)

        # y_pre = xgb_host.predict_prob(X_host)
        y_train_pre = xgb_host.predict_prob(X_host)
        y_train_pre.to_csv(predict_file_path)
        y_train_true = Y
        # Y_true = {"train": y_train_true, "test": y_true}
        # Y_pre = {"train": y_train_pre, "test": y_pre}
        Y_true = {"train": y_train_true}
        Y_pre = {"train": y_train_pre}
        if eva_type == 'regression':
            Regression_eva.get_result(Y_true, Y_pre, indicator_file_path)
        elif eva_type == 'classification':
            Classification_eva.get_result(Y_true, Y_pre, indicator_file_path)

    elif cry_pri == "plaintext":
        xgb_host = XGB_HOST(n_estimators=num_tree, max_depth=max_depth, reg_lambda=1,
                            sid=0, min_child_weight=1, objective='linear', proxy_server=proxy_server, proxy_client_guest=proxy_client_guest)
        # channel.recv()
        y_hat = np.array([0.5] * Y.shape[0])
        for t in range(xgb_host.n_estimators):
            logger.info("Begin to trian tree {}.".format(t))

            xgb_host.record = 0
            xgb_host.lookup_table = pd.DataFrame(
                columns=['record_id', 'feature_id', 'threshold_value'])
            f_t = pd.Series([0] * Y.shape[0])
            gh = xgb_host.get_gh(y_hat, Y)
            # xgb_host.channel.send(gh)
            proxy_client_guest.Remote(gh, 'gh')
            # GH_guest = xgb_host.channel.recv()
            GH_guest = proxy_server.Get('gh_sum')
            xgb_host.tree_structure[t + 1], f_t = xgb_host.xgb_tree(X_host, GH_guest, gh, f_t, 0)  # noqa
            xgb_host.lookup_table_sum[t + 1] = xgb_host.lookup_table
            y_hat = y_hat + xgb_host.learning_rate * f_t

            logger.info("Finish to trian tree {}.".format(t))

        predict_file_path = ph.context.Context.get_predict_file_path()
        indicator_file_path = ph.context.Context.get_indicator_file_path()
        model_file_path = ph.context.Context.get_model_file_path()
        lookup_file_path = ph.context.Context.get_host_lookup_file_path()

        with open(model_file_path, 'wb') as fm:
            pickle.dump(xgb_host.tree_structure, fm)
        with open(lookup_file_path, 'wb') as fl:
            pickle.dump(xgb_host.lookup_table_sum, fl)
        # y_pre = xgb_host.predict_prob(data_test)
        y_pre = xgb_host.predict_prob(X_host)
        print("==========", Y, y_pre)
        # if eva_type == 'regression':
        #     Regression_eva.get_result(Y, y_pre, indicator_file_path)
        # elif eva_type == 'classification':
        #     Classification_eva.get_result(Y, y_pre, indicator_file_path)

        # xgb_host.predict_prob(X_host).to_csv(predict_file_path)
    proxy_server.StopRecvLoop()
    end = time.time()
    logger.info("lasting time for xgb %s".format(end-start))


@ ph.context.function(role='guest', protocol='xgboost', datasets=['guest_dataset'], port='9000', task_type="regression")
def xgb_guest_logic(cry_pri="paillier"):
    # def xgb_guest_logic(cry_pri="plaintext"):
    print("start xgb guest logic...")
    # ios = IOService()
    # logger.info(ph.context.Context.dataset_map)
    # logger.info(ph.context.Context.node_addr_map)
    # logger.info(ph.context.Context.role_nodeid_map)
    # logger.info(ph.context.Context.params_map)
    role_node_map = ph.context.Context.get_role_node_map()
    node_addr_map = ph.context.Context.get_node_addr_map()
    dataset_map = ph.context.Context.dataset_map

    logger.debug(
        "dataset_map {}".format(dataset_map))

    logger.debug(
        "role_nodeid_map {}".format(role_node_map))

    logger.debug(
        "node_addr_map {}".format(node_addr_map))

    data_key = list(dataset_map.keys())[0]

    eva_type = ph.context.Context.params_map.get("taskType", None)
    if eva_type is None:
        logger.warn(
            "taskType is not specified, set to default value 'regression'.")
        eva_type = "regression"

    eva_type = eva_type.lower()
    if eva_type != "classification" and eva_type != "regression":
        logger.error(
            "Invalid value of taskType, possible value is 'regression', 'classification'.")
        return

    logger.info("Current task type is {}.".format(eva_type))

    if len(role_node_map["host"]) != 1:
        logger.error("Current node of host party: {}".format(
            role_node_map["host"]))
        logger.error("In hetero XGB, only dataset of host party has label,"
                     "so host party must have one, make sure it.")
        return

    guest_nodes = role_node_map["guest"]
    guest_port = node_addr_map[guest_nodes[0]].split(":")[1]
    proxy_server = ServerChannelProxy(guest_port)
    proxy_server.StartRecvLoop()
    logger.debug("Create server proxy for guest, port {}.".format(guest_port))

    host_nodes = role_node_map["host"]
    host_ip, host_port = node_addr_map[host_nodes[0]].split(":")

    proxy_client_host = ClientChannelProxy(host_ip, host_port,
                                           "host")
    data = ph.dataset.read(dataset_key=data_key).df_data
    X_guest = data

    if cry_pri == "paillier":
        xgb_guest = XGB_GUEST_EN(n_estimators=num_tree, max_depth=max_depth, reg_lambda=1, min_child_weight=1,
                                 objective='linear',
                                 sid=1, proxy_server=proxy_server, proxy_client_host=proxy_client_host)  # noqa
        # channel.send(b'guest ready')
        # pub = xgb_guest.channel.recv()
        pub = proxy_server.Get('xgb_pub')
        # xgb_guest.channel.send(b'recved pub')

        for t in range(xgb_guest.n_estimators):
            xgb_guest.record = 0
            xgb_guest.lookup_table = pd.DataFrame(
                columns=['record_id', 'feature_id', 'threshold_value'])
            # gh_host = xgb_guest.channel.recv()
            gh_host = proxy_server.Get('gh_en')
            X_guest_gh = pd.concat([X_guest, gh_host], axis=1)
            print("X_guest_gh: ", X_guest_gh.head)
            gh_sum = xgb_guest.get_GH(X_guest_gh, pub)
            # xgb_guest.channel.send(gh_sum)
            print("gh_sum: ", gh_sum)
            proxy_client_host.Remote(gh_sum, "gh_sum")
            xgb_guest.cart_tree(X_guest_gh, 0, pub)
            xgb_guest.lookup_table_sum[t + 1] = xgb_guest.lookup_table

        lookup_file_path = ph.context.Context.get_guest_lookup_file_path()

        with open(lookup_file_path, 'wb') as fl:
            pickle.dump(xgb_guest.lookup_table_sum, fl)
        # xgb_guest.predict(data_test)
        # xgb_guest.predict(X_guest)
    elif cry_pri == "plaintext":
        xgb_guest = XGB_GUEST(n_estimators=num_tree, max_depth=max_depth, reg_lambda=1, min_child_weight=1,
                              objective='linear',
                              sid=1, proxy_server=proxy_server, proxy_client_host=proxy_client_host)  # noqa
        # channel.send(b'guest ready')
        for t in range(xgb_guest.n_estimators):
            xgb_guest.record = 0
            xgb_guest.lookup_table = pd.DataFrame(
                columns=['record_id', 'feature_id', 'threshold_value'])
            # gh_host = xgb_guest.channel.recv()
            gh_host = proxy_server.Get('gh')
            X_guest_gh = pd.concat([X_guest, gh_host], axis=1)
            print(X_guest_gh)
            gh_sum = xgb_guest.get_GH(X_guest_gh)
            # xgb_guest.channel.send(gh_sum)
            proxy_client_host.Remote(gh_sum, 'gh_sum')
            xgb_guest.cart_tree(X_guest_gh, 0)
            xgb_guest.lookup_table_sum[t + 1] = xgb_guest.lookup_table

        lookup_file_path = ph.context.Context.get_guest_lookup_file_path()

        with open(lookup_file_path, 'wb') as fl:
            pickle.dump(xgb_guest.lookup_table_sum, fl)
        xgb_guest.predict(X_guest)
        # xgb_guest.predict(X_guest)
    proxy_server.StopRecvLoop()
