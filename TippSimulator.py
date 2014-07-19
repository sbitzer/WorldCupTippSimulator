# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 16:47:15 2014

@author: Sebastian Bitzer (official@sbitzer.eu)
"""

import csv
import numpy as np
import re
from scipy.stats import poisson

import matplotlib.pyplot as plt


def parseScore(scstr):
    match = re.match('(\d+):(\d+)(?: a.e.t.)?(?: \([:,\d\s]+\))?(?: (\d+):(\d+) PSO)?', scstr)
    score = [int(match.group(1)), int(match.group(2))]
    
    if match.lastindex > 2:
        penscore = [int(match.group(3)), int(match.group(4))]
        penalties = True
    else:
        penscore = score
        penalties = False
        
    return [score, penscore, penalties]


def readWCHistory(fnames=['WMresultate_2006.txt', 'WMresultate_2010.txt']):
    
    worldcups = [];
    
    for fn in fnames:
        with open(fn, 'r') as resfile:
            games = [];
            
            resreader = csv.reader(resfile, delimiter = ';')
            for row in resreader:
                if row[0].isdigit():
                    game = dict()
                    game['nr'] = int(row[0])
                    score = parseScore(row[5])
                    game['score'] = score[0]
                    game['penscore'] = score[1]
                    game['penalties'] = score[2]
                    
                    games.append(game)
                    
            worldcups.append(games)
            
    
    return worldcups


class ScoreDist(object):
    """A distribution over football scores"""
    
    
    def __init__(self, history = [], goalmax = 9, draws = True):
        # are draws allowed?
        self.draws = draws
        
        # define number of discrete scores that you want to look at
        self.goalmax = goalmax
        self.nscores = np.arange(1, self.goalmax + 2).sum()
        
        # define corresponding scores
        self.scores = np.zeros((self.nscores, 2))
        
        ind = 0
        for i in range(0, self.goalmax + 1):
            for j in range(i + 1):
                self.scores[ind, :] = [i, j]
                ind = ind + 1
        
        # probability that the score is swapped (away team wins, no effect for draw)
        self.swapprob = 0.5
        
        if len(history) == 0:
            # average number of goals scored in a match in world cups since 1994
            avgoals = (141 + 171 + 161 + 147 + 143) / (64 * 5)
            self.initPoisson(avgoals)
        else:
            self.initHistory(history)
            
        if not draws:
            drawinds = np.hstack([0, np.arange(2, 9+2)]).cumsum()
            self.probs[drawinds] = 0
            self.probs = self.probs / self.probs.sum()
            
        self.cdf = self.probs.cumsum()
        
        
    def initPoisson(self, avgoals = 2):
        # get probabilities over number of goals in a match according to 
        # Poisson distribution
        goalprobs = poisson.pmf( range(self.goalmax * 2 + 1), avgoals )
        
        ngoals = self.scores.sum(1)
        
        # determine score probabilities from Poisson distribution
        self.probs = np.ones((self.nscores))
        for g in range(self.goalmax * 2 + 1):
            # find number of scores with g goals
            inds = ngoals == g
            nsc = inds.sum()
            
            # set probability of scores as uniform between scores with the same
            # number of goals
            self.probs[inds] = goalprobs[g] / nsc
        
        
    def initHistory(self, scores):
        nscores = scores.shape[0]
        
        # count the occurrences of different scores
        sccnts = np.zeros((self.nscores))
        for match in range(nscores):
            sc = scores[match, :]
            sc.sort()
            ind = self.getScoreind(sc[::-1])
            sccnts[ind] = sccnts[ind] + 1
            
        # add a very small proportion which reduces for larger number of goals
        # to give all scores a non-zero probability
        sccnts = sccnts + 0.001 ** (self.scores.sum(1) + 1)
        
        # normalise
        self.probs = sccnts / sccnts.sum()

        
    def getScoreind(self, score):
        return np.arange(1, score[0] + 1).sum() + score[1]
        
    def getFullDist(self):
        winind = np.flatnonzero( (self.scores[:, 0] - self.scores[:, 1]) != 0 )
        scores = np.vstack( (self.scores, self.scores[winind, ::-1]) )
        probs = np.hstack( (self.probs, self.probs[winind]) )
        probs[winind] = probs[winind] * self.swapprob
        probs[self.nscores:] = probs[self.nscores:] * self.swapprob
        
        return scores, probs
        

class TippSimulator(object):
    """Implements a model which can predict tips for football games 
       of a world cup based on historical data"""
    
    def __init__(self, history = [], goalmax = 9):
        # first stage (groups)
        self.stage1dist = ScoreDist(self.extractScores(history, np.arange(1, 49)), 
                                    goalmax)
        
        # second stage (knockout)
        self.stage2dist = ScoreDist(self.extractScores(history, np.arange(49, 65)), 
                                    goalmax, draws = False)
        
        
    def extractScores(self, history, matchnr):
        nwcs = len(history)
        if nwcs == 0:
            return []
            
        scores = []
        for wc in history:
            for match in wc:
                if np.any(match['nr'] == matchnr):
                    scores.append(match['penscore'])
        
        return np.array(scores)
                
        
    def generateScores(self, ScD, N):
        # sample indices of scores
        scorecnts = np.random.multinomial(N, ScD.probs)
        inds = np.zeros((N), dtype=np.int8)
        cnt = 0
        for i in range(ScD.nscores):
            inds[ cnt : cnt + scorecnts[i] ] = i
            cnt = cnt + scorecnts[i]
        
        # determine winners
        awayinds = np.random.rand(N) > ScD.swapprob
        
        # transform indices to scores
        scores = ScD.scores[ inds, :]
        scores[awayinds, :] = np.vstack([scores[awayinds, 1], 
                                         scores[awayinds, 0]]).T
        
        # shuffle
        np.random.shuffle(scores)
        
        return scores
        
        
    def sampleWorldcup(self, N):
        """samples scores for a whole world cup using stage 1 and 2 models"""
        
        # sample N * 48 stage 1 scores
        S1scores = self.generateScores(self.stage1dist, N * 48)
        # sample N * 16 stage 2 scores
        S2scores = self.generateScores(self.stage2dist, N * 16)
        
        # reshape into [match, goals, numer of samples] shape
        # note that this maintains the samled scores, but not their order
        S12scores = np.vstack(( S1scores.reshape((48, 2*N), order='C'), 
                                S2scores.reshape((16, 2*N), order='C') ))
        
        return S12scores.reshape( (64, 2, N), order='F')    
        
        
class WC2014Tippspiel(object):
    """computes Tippspiel points for tipps made for the world cup 2014"""
    
    def __init__(self, prule = np.array([2, 3, 4])):
        # 0 - correct team, 1 - correct goal difference, 2 - correct score
        self.prule = prule;
        
        # read scores of matches in order of kick-off time
        self.readWC2014scores()
        
        
    def readWC2014scores(self, fname = 'WMresultate_2014_kicktipp.txt'):
        with open(fname, 'r') as resfile:
            self.scores = []
            
            resreader = csv.reader(resfile, delimiter = ';')
            for row in resreader:
                if row[0] == 'WM':
                    match = re.match(' (\d+):(\d+)[nEnV]*', row[6])
                    if match == None:
                        break
                    else:
                        score = [int(match.group(1)), int(match.group(2))]
                        self.scores.append(score)

            self.nscores = len(self.scores)
            self.scores = np.array(self.scores)
            self.scorediffs = self.scores[:, 0] - self.scores[:, 1]
            self.wins = np.sign(self.scorediffs)
            self.wininds =  self.wins != 0.
            
            
    def readWC2014tipps(self, fname='WM2014_tipps.txt'):
        names = []
        scores = []
        
        with open(fname, 'r') as tipfile:
            tipreader = csv.reader(tipfile, delimiter = ';')
            for row in tipreader:
                names.append(row[0])
                scs = []
                for scstr in row[1:]:
                    match = re.match(' ?(\d+):(\d+)', scstr)
                    scs.append([int(match.group(1)), int(match.group(2))])
                
                scores.append(scs)
                
        return names, np.array(scores).transpose(1, 2, 0)
        
            
    def compWCPoints(self, predscore):
        """computes points for one or more full sets of world cup predictions"""
        
        # number of predictions (of full world cup)
        if len(predscore.shape) == 3:
            npred = predscore.shape[2]
        else:
            npred = 1
            predscore = np.reshape(predscore, predscore.shape + (1,))
        
        points = np.zeros((self.nscores, npred))
        
        for p in range(npred):
            scdiffs = (  predscore[:self.nscores, 0, p] 
                       - predscore[:self.nscores, 1, p] )
            
            # get scores which had the correct winner
            pinds = np.flatnonzero(  (np.sign(scdiffs) == self.wins) 
                                   & self.wininds)
            
            # select correct score difference for a won game
            dinds = np.flatnonzero( scdiffs[pinds] == self.scorediffs[pinds] )
            # record corresponding points
            points[pinds[dinds], p] = self.prule[1]
            
            # select correct score
            scerror = (  predscore[pinds[dinds], :, p] 
                       - self.scores[pinds[dinds], :] )
            sinds = scerror.sum(1) == 0
            # record corresponding points
            points[pinds[dinds[sinds]], p] = self.prule[2]
            
            # select correct winner (all pinds which don't have a correct score difference)
            winds = np.setdiff1d(range(len(pinds)), dinds, assume_unique=True)
            # record corresponding points
            points[pinds[winds], p] = self.prule[0]
            
            # now do again for correctly predicted draws
            pinds = np.flatnonzero(  (np.sign(scdiffs) == self.wins) 
                                   & ~self.wininds)
            if len(pinds) > 0:
                sinds = (predscore[pinds, 0, p] - self.scores[pinds, 0]) == 0
                points[pinds[sinds], p] = self.prule[2]
                points[pinds[~sinds], p] = self.prule[0]
            
        return points
        
    def compPoints(self, mind, predscores):
        """compute points for a set of scores predicted for a single match"""
        
        scdiffs = predscores[:, 0] - predscores[:, 1]
        
        # indices of the predicted scores that had the correct outcome 
        # (draw or win for the team that really won)
        pinds = np.flatnonzero( np.sign(scdiffs) == self.wins[mind] )
        
        scdiffs = scdiffs[pinds]
        
        # absolute error in scores
        scerror = np.abs(predscores[pinds, :] - self.scores[mind, :]).sum(1)
        
        # initialise points
        points = np.zeros(predscores.shape[0])
        
        # points for correctly predicted score
        points[pinds[scerror == 0]] = self.prule[2]
        
        # if the match had a winner
        if self.wininds[mind]:
            points[pinds[ (scerror != 0) & 
                          (scdiffs == self.scorediffs[mind]) ]] = self.prule[1]
            points[pinds[ (scerror != 0) & 
                          (scdiffs != self.scorediffs[mind]) ]] = self.prule[0]
            
        else:
            points[pinds[ scerror != 0 ]] = self.prule[0]
            
        return points
        
        
    def compPointDist(self, sim):
        """computes a point distribution from the score distributions stored
           in the tip simulator"""
        
        # initialise point distribution
        maxP = 64 * self.prule[2]
        Pdist = np.zeros((maxP+1, 65))
        Pdist[0, 0] = 1
        
        dscores, dprobs = sim.stage1dist.getFullDist()
        
        # for all WC2014 matches
        for m in range(1, self.nscores+1):
            # the 49th match is the first in the second stage
            if m == 48:
                dscores, dprobs = sim.stage2dist.getFullDist()
            
            # compute points for all scores in the scoredist
            scPoints = self.compPoints(m-1, dscores)
            
            # get distribution of points for this match and convolve with old
            # point distribution to get new point distribution
            pproball = 0
            for p in self.prule:
                pprob = dprobs[scPoints == p].sum()
                for P in range(p, maxP+1):
                    Pdist[P, m] = Pdist[P, m] + Pdist[P-p, m-1] * pprob
                
                pproball = pproball + pprob
                
            Pdist[:, m] = Pdist[:, m] + Pdist[:, m-1] * (1 - pproball)
            
        return Pdist[:, 1:]


def plotPersonPoints(Ptips, names, maxp=120):
    nm, nt = Ptips.shape
    np = len(names)
    
    cols = ['0.3', '0.7']
    
    plt.figure(figsize=(8, 4))
    
    for tip in range(nt):
        plt.plot(range(1, nm+1), Ptips[:, tip], linewidth=2, label=names[tip],
                 c=cols[tip])
    
    plt.xlim(1, nm)
    plt.ylim(0, maxp)
    plt.xlabel('Spielnummer')
    plt.ylabel('Punkte')
    leg = plt.legend(loc=2, frameon=False)
    
    texts = leg.get_texts()
    for p in range(np):
        texts[p].set_color(cols[p])
    
    return leg
    

def plotTippPointsWithDist(Pdist, Ptips, names):
    maxp = 120
    nm = Ptips.shape[0]
    
    plotPersonPoints(Ptips, names, maxp)
    
    cm = plt.get_cmap('hot_r')
    im1 = plt.pcolormesh(np.arange(1, nm+1), np.arange(maxp+1), 
                         Pdist[:maxp+1, :nm], vmin=0, vmax=0.05, cmap=cm, 
                         shading='gouraud')
    
    cbar = plt.colorbar(im1)
    cbar.set_label('Wahrscheinlichkeit', rotation=270)
    
    plt.show()
    
    
def plotTippPointsWithCDF(Pdist, Ptips, names, pval=0.01):
    maxp = 120
    nm = Ptips.shape[0]
    
    plotPersonPoints(Ptips, names, maxp)
    
    dcols = ('#000000', '#FF0000', '#FFCC00')
    
    Pcdf = Pdist[:maxp+1, :nm].cumsum(0)
    
    upP = np.zeros(nm)
    botP = np.zeros(nm)
    for m in range(nm):
        pinds = np.flatnonzero(Pcdf[:, m] > 1 - pval)
        upP[m] = pinds[0]
        
        pinds = np.flatnonzero(Pcdf[:, m] < pval)
        if pinds.size > 0:
            botP[m] = pinds[-1]
    
    a = 1
    plt.fill_between(range(1, nm+1), upP, maxp, facecolor=dcols[0], 
                     edgecolor='none', interpolate=False, alpha=a)
    plt.fill_between(range(1, nm+1), botP, upP, facecolor=dcols[1], 
                     edgecolor=dcols[1], interpolate=False, alpha=a)
    plt.fill_between(range(1, nm+1), 0, botP, facecolor=dcols[2], 
                     edgecolor='none', interpolate=False, alpha=a)
    
    plt.show()
    
    
    print 'points to be above pval =', pval, ':', upP[-1]
    

#if __name__ == "__main__":
#    worldcups = readWCHistory()
#    sim = TippSimulator(worldcups)
#    tip = WC2014Tippspiel()
#    
#    # sample
##    N = 10000
##    S = sim.sampleWorldcup(N)
##    points = tip.compWCPoints(S)
##    pointscum = points.cumsum(0)
##    PSdist = np.empty((64 * tip.prule[2] + 1, pointscum.shape[0]))
##    bins = range(64 * tip.prule[2]+2)
##    for m in range(pointscum.shape[0]):
##        PSdist[:, m], _ = np.histogram(pointscum[m, :], bins, density=True)
#    
#    # analytic point distribution based on score model
#    Pdist = tip.compPointDist(sim)
#    
#    # read tipp scores
#    names, tscores = tip.readWC2014tipps()
#    
#    # compute points for each tipp
#    tpoints = tip.compWCPoints(tscores)
#    
#    # plot tipp points together with point distribution
#    plotTippPointsWithDist(Pdist, tpoints.cumsum(0), names)
#    
#    plotTippPointsWithCDF(Pdist, tpoints.cumsum(0), names)