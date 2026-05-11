import numpy as np
import numba
from math import floor

# 55d is same as c for execs, but addes in the mutation dynamic for for the model without execs

# put functions in this cell

@numba.jit(nopython=True, fastmath=False, parallel=False)
def getalphastyped0(popt, execsruns, final_execurns, agreeruns, oppositiondisruns, type0, type0opposition, type0agree, rdex, multidimsignalscount0, runsmultidimsignalscount0, sigmultipliers, population_size, final_sigurns_typed, numsignals_perdim, sigdimensions, runs, numtypes, numsignals):
    # first lets figure out which signal is alpha for each dimension
    # also adding in some counts for what dimensions execs attend to and making the most attended to dimensions by type0 the alpha dimension
    alphasignals= np.zeros((sigdimensions), dtype=np.int64)
    alphaexecs = np.zeros((numtypes, sigdimensions), dtype=np.int64)
    totalexecs = np.zeros((numtypes, sigdimensions), dtype=np.int64)
    for dimdex0 in range(0, sigdimensions):
        dimalpha = np.zeros((numsignals_perdim[dimdex0]))
        for popdex0 in range(0, population_size):
            currenttype = popt[popdex0]
            totalexecs[currenttype][dimdex0] += 1
            if final_execurns[rdex][popdex0][dimdex0] == 1:
                alphaexecs[currenttype][dimdex0] += 1
            if currenttype == type0:
                t0agsig = np.argmax(final_sigurns_typed[dimdex0][popdex0])
                dimalpha[t0agsig] += 1
        alpha4thisdim = np.argmax(dimalpha)
        alphasignals[dimdex0] = alpha4thisdim
    # now we know which signals are alpha
    execsindex = np.argsort(alphaexecs[type0])
    alphaexecsthresh = np.divide(alphaexecs, totalexecs)
    for typedex4 in range(0, numtypes):
        for dimdex4 in range(0, sigdimensions):
            if alphaexecsthresh[typedex4][dimdex4] <= 0.5:
                alphaexecsthresh[typedex4][dimdex4] = 0
            else:
                alphaexecsthresh[typedex4][dimdex4] = 1
    alphaexecsthresh[:] = alphaexecsthresh[:, execsindex]
    # now we know execs
    execsruns[rdex] = alphaexecsthresh
    
    # lets count the number of type opposition agents that have maximay dissimilar signal from type 0
    totalopposition = np.zeros((len(type0opposition)), dtype=np.int64)
    oppositiondis = np.zeros((len(type0opposition)), dtype=np.int64)
    for popdex0 in range(0, population_size):
        dimalpha = np.zeros((numsignals_perdim[dimdex0]))
        if (popt[popdex0] == type0opposition).any():
            optype = popt[popdex0]
            opdex = np.searchsorted(type0opposition, optype)
            totalopposition[opdex] += 1
            zero = 0
            for dimdex0 in range(0, sigdimensions): 
                if np.argmax(final_sigurns_typed[dimdex0][popdex0]) == alphasignals[dimdex0]:
                    zero += 1
            if zero == 0:
                oppositiondis[opdex] += 1
    for optypedex in range(0, len(type0opposition)):
        if (oppositiondis[optypedex]/totalopposition[optypedex]) > 0.5:
            oppositiondisruns[optypedex][rdex] = 1
    # lets count the number of type agree agents that have maximay dissimilar signal from type 0
    totalagree = np.zeros((len(type0agree)), dtype=np.int64)
    agree = np.zeros((len(type0agree)), dtype=np.int64)
    for popdex0 in range(0, population_size):
        dimalpha = np.zeros((numsignals_perdim[dimdex0]))
        if (popt[popdex0] == type0agree).any():
            agtype = popt[popdex0]
            agdex = np.searchsorted(type0agree, agtype)
            totalagree[agdex] += 1
            zero = 0
            for dimdex0 in range(0, sigdimensions): 
                if np.argmax(final_sigurns_typed[dimdex0][popdex0]) == alphasignals[dimdex0]:
                    zero += 1
            if zero > 0:
                agree[opdex] += 1
    for agtypedex in range(0, len(type0agree)):
        if (agree[agtypedex]/totalagree[agtypedex]) > 0.5:
            agreeruns[agtypedex][rdex] = 1
    
    # ughhhh this current version is only going to work for systems with only two signals in each dimension
    # fixing this by replacing the 0 signal with the alpha signals previous value
    for popdex1 in range(0, population_size):
        sigdexcount = 0
        for dimdex1 in range(0, sigdimensions):
            if np.argmax(final_sigurns_typed[dimdex1][popdex1]) == alphasignals[dimdex1]: 
                fakesig = 0
            elif np.argmax(final_sigurns_typed[dimdex1][popdex1]) == 0: # replacing the 0 signal with the previous alpha signal value
                fakesig = alphasignals[dimdex1]
            else:
                fakesig = np.argmax(final_sigurns_typed[dimdex1][popdex1])
            sigdexcount += fakesig*sigmultipliers[dimdex1]
        agtype = floor(popt[popdex1])
        sigdex = floor(sigdexcount)
        runsmultidimsignalscount0[rdex][agtype][sigdex] += 1
    for typesdex in range(0, numtypes):
        mostcommonbytype = np.argmax(runsmultidimsignalscount0[rdex][typesdex])
        multidimsignalscount0[typesdex][mostcommonbytype] += 1
    return multidimsignalscount0, runsmultidimsignalscount0, oppositiondisruns, agreeruns, execsruns

# function to cohere for aggregation such that type0's most common signal is always alpha, includes count of how frequently type0's opposition is maximally dissimilar
# lets find a way to allow typ0opposition and type0agree to be lists of potentially multiple types
def getalphasignaling(popt, final_execurns, type0, type0opposition, type0agree, sigmultipliers, population_size, final_sigurns, numsignals_perdim, sigdimensions, runs, numtypes, numsignals):
    multidimsignalscount0 = np.zeros((numtypes, numsignals))
    runsmultidimsignalscount0 = np.zeros((runs, numtypes, numsignals))
    # lets have a variable note the runs for which more than 50% of type opposition agents had a signal that was maximally dissimilar from type 0
    oppositiondisruns = np.zeros((len(type0opposition), runs), dtype=np.int64)
    agreeruns = np.zeros((len(type0agree), runs), dtype=np.int64)
    execsruns = np.zeros((runs, numtypes, sigdimensions))

    for rdex in range(0, runs):
        final_sigurns_typed = numba.typed.List(final_sigurns[rdex])
        multidimsignalscount0, runsmultidimsignalscount0, oppositiondisruns, agreeruns, execsruns = getalphastyped0(popt, execsruns, final_execurns, agreeruns, oppositiondisruns, type0, type0opposition, type0agree, rdex, multidimsignalscount0, runsmultidimsignalscount0, sigmultipliers, population_size, final_sigurns_typed, numsignals_perdim, sigdimensions, runs, numtypes, numsignals)

    multidimsignalscount0 = multidimsignalscount0/runs
    
    meanexecsruns = np.sum(execsruns, axis = 0)
    meanexecsruns = meanexecsruns/runs
    
    # lets also report (1) average dimensions attended to and (2) for each type average number of dimensions in which they agree with type0
    return multidimsignalscount0, runsmultidimsignalscount0, oppositiondisruns, agreeruns, execsruns, meanexecsruns



# function from population size and percentage of identity types to array giving the type of each individual in the population
def population_array(population_sizef, percent_agents_per_typef):
    count0000 = 0
    cumsum_papt = np.cumsum(percent_agents_per_typef)
    poptypes = np.zeros(population_sizef, dtype=np.int64)
    for idx0000 in range(0, population_sizef):
        if idx0000 >= (population_sizef*cumsum_papt[count0000]):
            count0000 += 1
        poptypes[idx0000] = count0000
    return poptypes


# function generating weights for probability of a pairing between agents
# this function was taken from group_id0001 model and may need variable names changed.
# popid should now be given by the social signal chosen on a given timestep
@numba.jit(nopython=True, fastmath=False, parallel=False)
def connections_init(population_sizef, popidf, base_connection_weightsf, homophily_factorf):
    conweightsf = np.zeros((population_sizef, population_sizef), dtype=np.int64)
    for idx1000 in range(0, population_sizef):
        for idx1001 in range(1, population_sizef):
            idx1001b = (idx1000+idx1001)%population_sizef
            dist = np.sum(popidf[idx1000] == popidf[idx1001b])
            conweightsf[idx1000][idx1001b] = (base_connection_weightsf[dist])**homophily_factorf
    return conweightsf




# function for social signal draws
@numba.jit(nopython=True, fastmath=False, parallel=False)
def social_draws(sigurnsf, population_sizef, sigdimensionsf, rngf, epsilonf):
    sigdraws = np.zeros((population_sizef, sigdimensionsf), dtype=np.int64)
    for idx8000 in range(0, sigdimensionsf):
        for idx8001 in range(0, population_sizef):
            # cs = current signaler
            csurn = sigurnsf[idx8000][idx8001]
            cscumsum = np.cumsum(csurn)
            csrand = (rngf.random()+epsilonf)*cscumsum[-1]
            csdraw = np.zeros(len(cscumsum))
            csdraw[cscumsum<csrand] = 1
            sigdraws[idx8001][idx8000] = np.sum(csdraw)
    return sigdraws

#function to take conweights, a permutation of pop size and random uniform of pop size and return our weighted random pairings
#function to take conweights, a permutation of pop size and random uniform of pop size and return our weighted random pairings
@numba.jit(nopython=True)
def pairings(randperm01f, rngf, population_sizef, conweightsf, epsilonf):
    halfpop = population_sizef//2
    pairsf = np.zeros((halfpop, 2), dtype=np.int64)
    conf = conweightsf.copy()
    idx2000 = 0
    count2000 = 0
    while idx2000 < halfpop:
        idx2001 = randperm01f[count2000]
        agweight = (conf[idx2001]).copy()
        agsum = np.sum(agweight)
        if agsum != 0:
            agpick = agsum*(epsilonf+rngf.random())
            agweight = np.cumsum(agweight)
            agweight[agweight > agpick] = -1
            agweight[agweight != -1] = 1
            agweight[agweight == -1] = 0
            pairedag = floor(np.sum(agweight))
            pairsf[idx2000] = [idx2001, pairedag]
            conf[idx2001] = 0
            conf[:, idx2001] = 0
            conf[pairedag] = 0
            conf[:, pairedag] = 0
            
            idx2000 += 1
        count2000 += 1
    
    return pairsf

@numba.jit(nopython=True, fastmath=False, parallel=False)
def receiver_draws(population_sizef, pairsf, rngf, recurnsf, sigdraws, sigmultipliersf, sigdimensionsf, epsilonf):
    recdraws = np.zeros((population_sizef), dtype=np.int64)
    halfpop = population_sizef//2
    for idx4000 in range(0, halfpop):
        arec = pairsf[idx4000][0]
        brec = pairsf[idx4000][1]
        aidx = 0
        bidx = 0
        for idx4001 in range(0, sigdimensionsf):
            aidx += sigdraws[brec][idx4001]*sigmultipliersf[idx4001]
            bidx += sigdraws[arec][idx4001]*sigmultipliersf[idx4001]
        aurn = recurnsf[arec][floor(aidx)]
        burn = recurnsf[brec][floor(bidx)]
        acumsum = np.cumsum(aurn)
        bcumsum = np.cumsum(burn)
        arand = (rngf.random()+epsilonf)*acumsum[-1]
        brand = (rngf.random()+epsilonf)*bcumsum[-1]
        adraw = np.zeros(len(acumsum))
        bdraw = np.zeros(len(bcumsum))
        adraw[acumsum<arand] = 1
        bdraw[bcumsum<brand] = 1
        recdraws[arec] = np.sum(adraw)
        recdraws[brec] = np.sum(bdraw)
        
    return recdraws

@numba.jit(nopython=True, fastmath=False, parallel=False)
def random2picks(randoms4pick, sigdimensionsf, sigurnsf, recurnsf, population_sizef, agentprofilepicks, profile_length, profilecaps, numsignalsf):
    for idx8000 in range(0, sigdimensionsf):
        siglen = profilecaps[idx8000]
        for idx8001 in range(0, population_sizef):
            # cs = current signaler
            csurn = sigurnsf[idx8000][idx8001]
            cscumsum = np.cumsum(csurn)
            csrand = (randoms4pick[idx8001][idx8000])*cscumsum[-1]
            csdraw = np.zeros(siglen)
            csdraw[cscumsum<csrand] = 1
            agentprofilepicks[idx8001][idx8000] = np.sum(csdraw)
    reclen = profilecaps[-1]
    for idx8002 in range(0, population_sizef):
        for idx8003 in range(0, numsignalsf):
            recprofiledex = sigdimensionsf+idx8003
            crurn = recurnsf[idx8002][idx8003]
            crcumsum = np.cumsum(crurn)
            crrand = (randoms4pick[idx8002][recprofiledex])*crcumsum[-1]
            crdraw = np.zeros(reclen)
            crdraw[crcumsum<crrand] = 1
            agentprofilepicks[idx8002][recprofiledex] = np.sum(crdraw)
    return agentprofilepicks
            
    
@numba.jit(nopython=True, fastmath=False, parallel=False)
def greetings_check_success(sigdimensionsf, sigmultipliersf, reinforcementf, punishmentf, population_sizef, sigurnsf, pairsf, sigdraws, recurnsf, recdraws, poptf):
    halfpop = population_sizef//2
    for idx5000 in range(0, halfpop):
        adex = pairsf[idx5000][0]
        bdex = pairsf[idx5000][1]
        atype = poptf[adex]
        btype = poptf[bdex]
        adraw = recdraws[adex]
        bdraw = recdraws[bdex]
        
        aidx = 0
        bidx = 0
        for idx5001 in range(0, sigdimensionsf):
            aidx += sigdraws[bdex][idx5001]*sigmultipliersf[idx5001]
            bidx += sigdraws[adex][idx5001]*sigmultipliersf[idx5001]
        arecurn = floor(aidx)
        brecurn = floor(bidx)
        # check if a's action was correct and reinforce accordingly
        if adraw == btype: #success condition for a doing right greeting for b; then reinforcing a's receiver and b's senders
            recurnsf[adex][arecurn][adraw] += reinforcementf
            for idx5002 in range(0, sigdimensionsf):
                dimdraw = sigdraws[bdex][idx5002]
                sigurnsf[idx5002][bdex][dimdraw] += reinforcementf
        else: #then play was a filure so we need to punish
            recurnsf[adex][arecurn][adraw] += punishmentf
            if recurnsf[adex][arecurn][adraw] < 1:
                recurnsf[adex][arecurn][adraw] = 1
            for idx5002 in range(0, sigdimensionsf):
                dimdraw = sigdraws[bdex][idx5002]
                sigurnsf[idx5002][bdex][dimdraw] += punishmentf
                if sigurnsf[idx5002][bdex][dimdraw] < 1:
                    sigurnsf[idx5002][bdex][dimdraw] = 1
        # check if b's action was correct and reinforce accordingly
        if bdraw == atype: #success condition for b doing right greeting for a; then reinforcing b's receiver and a's senders
            recurnsf[bdex][brecurn][bdraw] += reinforcementf
            for idx5002 in range(0, sigdimensionsf):
                dimdraw = sigdraws[adex][idx5002]
                sigurnsf[idx5002][adex][dimdraw] += reinforcementf
        else: #then play was a filure so we need to punish
            recurnsf[bdex][brecurn][bdraw] += punishmentf
            if recurnsf[bdex][brecurn][bdraw] < 1:
                recurnsf[bdex][brecurn][bdraw] = 1
            for idx5002 in range(0, sigdimensionsf):
                dimdraw = sigdraws[adex][idx5002]
                sigurnsf[idx5002][adex][dimdraw] += punishmentf
                if sigurnsf[idx5002][adex][dimdraw] < 1:
                    sigurnsf[idx5002][adex][dimdraw] = 1
    return sigurnsf, recurnsf

@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBS_check_success(genBSpunishf, sigdimensionsf, sigmultipliersf, numactionsf, coordination_preferencesf, population_sizef, sigurnsf, pairsf, sigdraws, recurnsf, recdraws, poptf, rngf):
    halfpop = population_sizef//2
    for idx5000 in range(0, halfpop):
        adex = pairsf[idx5000][0]
        bdex = pairsf[idx5000][1]
        atype = poptf[adex]
        btype = poptf[bdex]
        adraw = recdraws[adex]
        bdraw = recdraws[bdex]
        
        aidx = 0
        bidx = 0
        for idx5001 in range(0, sigdimensionsf):
            aidx += sigdraws[bdex][idx5001]*sigmultipliersf[idx5001]
            bidx += sigdraws[adex][idx5001]*sigmultipliersf[idx5001]
        arecurn = floor(aidx)
        brecurn = floor(bidx)
        # check if a's action was correct and reinforce accordingly
        if adraw == bdraw: #success condition for a doing coordinating in genBS; then reinforcing a and b's urns
            recurnsf[adex][arecurn][adraw] += coordination_preferencesf[atype][adraw]
            recurnsf[bdex][brecurn][bdraw] += coordination_preferencesf[btype][bdraw]
#             socialcount = 0 # use this to see if agents sent the same social signal for all dimensions # ****************
            for idx5002 in range(0, sigdimensionsf):
                adimdraw = sigdraws[adex][idx5002]
                bdimdraw = sigdraws[bdex][idx5002]
                sigurnsf[idx5002][adex][adimdraw] += coordination_preferencesf[atype][adraw]
                sigurnsf[idx5002][bdex][bdimdraw] += coordination_preferencesf[btype][bdraw]
#                 if adimdraw != bdimdraw: # ****************
#                     socialcount = 1 # ****************
#             # adding in the social learning # ****************
#             if socialcount == 0: # ****************
#                 socialdraw = rngf.random() # ****************
#                 if socialdraw < social_learn_probf: # ****************
#                     recurnsf[adex] = recurnsf[bdex] # since its already random who is the a vs b in the pair, we cal always have a copy b # ****************
#                     for idx5003 in range(0, sigdimensionsf): # ****************
#                         sigurnsf[idx5003][adex] = sigurnsf[idx5003][bdex] # ****************
        # in the exec extension need to also include some code here for signal costs
        else:
            recurnsf[adex][arecurn][adraw] += genBSpunishf
            if recurnsf[adex][arecurn][adraw] < 1:
                recurnsf[adex][arecurn][adraw] = 1
            recurnsf[bdex][brecurn][bdraw] += genBSpunishf
            if recurnsf[bdex][brecurn][bdraw] < 1:
                recurnsf[bdex][brecurn][bdraw] = 1
            for idx5002 in range(0, sigdimensionsf):
                adimdraw = sigdraws[adex][idx5002]
                bdimdraw = sigdraws[bdex][idx5002]
                sigurnsf[idx5002][adex][adimdraw] += genBSpunishf
                if sigurnsf[idx5002][adex][adimdraw] < 1:
                    sigurnsf[idx5002][adex][adimdraw] = 1
                sigurnsf[idx5002][bdex][bdimdraw] += genBSpunishf
                if sigurnsf[idx5002][bdex][bdimdraw] < 1:
                    sigurnsf[idx5002][bdex][bdimdraw] = 1
    return sigurnsf, recurnsf

@numba.jit(nopython=True, fastmath=False, parallel=False)
def time_update(typed_time, signal_time, typed_time_norm, signal_time_norm, poptf, sigurnsf, recurnsf, population_sizef, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigmultipliersf, timedex):
    # copying here to remember the shape of typed_time and signal_time
#     typed_time = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.fp64)
#     signal_time = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.fp64)
    typed_count = np.zeros((numtypesf, numsignalsf, numactionsf), dtype=np.float64)
    signal_count = np.zeros((numsignalsf, numsignalsf, numactionsf), dtype=np.float64)
    typed_count_norm = np.zeros((numtypesf, numsignalsf, numactionsf), dtype=np.float64)
    signal_count_norm = np.zeros((numsignalsf, numsignalsf, numactionsf), dtype=np.float64)
    for idx6000 in range(0, population_sizef):
        agent_time_sig = 0# note that we're pulling the agents most likely social signal not neccessarily their signal from this timestep
        for idx6000b in range(0, sigdimensionsf):
            agent_time_sig += np.argmax(sigurnsf[idx6000b][idx6000])*sigmultipliersf[idx6000b]
        agent_time_sig = floor(agent_time_sig)
        agent_time_type = poptf[idx6000]
        for idx6001 in range(0, numsignalsf):
            for idx6002 in range(0, numactionsf):
                typed_count[agent_time_type][idx6001][idx6002] += recurnsf[idx6000][idx6001][idx6002] # note that counting this way means that more successful agents have stronger weight
                signal_count[agent_time_sig][idx6001][idx6002] += recurnsf[idx6000][idx6001][idx6002]
            # let's get the normalized statistic so each agent gets equal weight
            timenorm = 0
            for idx6002c in range(0, numactionsf):
                timenorm += recurnsf[idx6000][idx6001][idx6002c]
            for idx6002b in range(0, numactionsf):
                typed_count_norm[agent_time_type][idx6001][idx6002b] += (recurnsf[idx6000][idx6001][idx6002b]/timenorm)
                signal_count_norm[agent_time_sig][idx6001][idx6002b] += (recurnsf[idx6000][idx6001][idx6002b]/timenorm)
    # now lets normalize the time series data and put it in the time array from the count array
    for idx6003 in range(0, numtypesf):
        for idx6004 in range(0, numsignalsf):
            typedenom = np.sum(typed_count[idx6003][idx6004])
            for idx6005 in range(0, numactionsf):
                typed_time[idx6003][idx6004][idx6005][timedex] = typed_count[idx6003][idx6004][idx6005]/typedenom
    for idx6006 in range(0, numsignalsf):
        for idx6007 in range(0, numsignalsf):
            signaldenom = np.sum(signal_count[idx6006][idx6007])
            if signaldenom == 0:
                signaldenom = 1
            for idx6008 in range(0, numactionsf):
                signal_time[idx6006][idx6007][idx6008][timedex] = signal_count[idx6006][idx6007][idx6008]/signaldenom
    for idx6003d in range(0, numtypesf):
        for idx6004d in range(0, numsignalsf):
            typedenom = np.sum(typed_count_norm[idx6003d][idx6004d])
            for idx6005d in range(0, numactionsf):
                typed_time_norm[idx6003d][idx6004d][idx6005d][timedex] = typed_count_norm[idx6003d][idx6004d][idx6005d]/typedenom
    for idx6006d in range(0, numsignalsf):
        for idx6007d in range(0, numsignalsf):
            signaldenom = np.sum(signal_count_norm[idx6006d][idx6007d])
            if signaldenom == 0:
                signaldenom = 1
            for idx6008d in range(0, numactionsf):
                signal_time_norm[idx6006d][idx6007d][idx6008d][timedex] = signal_count_norm[idx6006d][idx6007d][idx6008d]/signaldenom
    
    return typed_time, signal_time, typed_time_norm, signal_time_norm

# function to populate the sender and receiver "urns" with random initial values
# also initializes an array that coorelates strategy profile with what the strategy is
@numba.jit(nopython=True, fastmath=False, parallel=False)
def replicator_initialize(rngf, repmultipliersf, numprofilesf, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigurnsf, recurnsf, population_sizef):
    sigperdim = np.zeros((sigdimensionsf), dtype=np.int64)
    for idx7000 in range(0, sigdimensionsf):
        sigperdim[idx7000] = len(sigurnsf[idx7000][0])
    for idx7001 in range(0, population_sizef):
        # generating signal urns to have all zeros, except a 1 in the location of the signal that is broadcasted
        for idx7002 in range(0, sigdimensionsf):
            csurn = sigurnsf[idx7002][idx7001]
            csdex = rngf.integers(0, len(csurn))
            cspopulate = np.zeros(len(csurn))
            cspopulate[csdex] = 1
            sigurnsf[idx7002][idx7001] = cspopulate
        # generating receiver urns similarly
        for idx7003 in range(0, numsignalsf):
            adex = rngf.integers(0, numactionsf)
            recpopulate = np.zeros(numactionsf)
            recpopulate[adex] = 1
            recurnsf[idx7001][idx7003] = recpopulate
    profiles_indexed = np.zeros((numprofilesf, sigdimensionsf+numsignalsf), dtype=np.int64)
    profile = np.zeros((sigdimensionsf+numsignalsf), dtype=np.int64)
    profilecaps = np.zeros((sigdimensionsf+numsignalsf), dtype=np.int64)
    for pcdex in range(0, sigdimensionsf+numsignalsf):
        if pcdex < sigdimensionsf:
            profilecaps[pcdex] = sigperdim[pcdex]
        else:
            profilecaps[pcdex] = numactionsf
            
    for idx7004 in range(1, numprofilesf):
        profile[0] = (profile[0]+1)%profilecaps[0]
        for idx7005 in range(1, sigdimensionsf+numsignalsf):
            zero = 0
            for idx7006 in range(0, idx7005):
                zero += profile[idx7006]
            if zero == 0:
                profile[idx7005] = (profile[idx7005]+1)%profilecaps[idx7005]
        profiles_indexed[idx7004] = profile
        # lets check that this was computed correctly
        profilecheck = 0
        for idx7007 in range(0, sigdimensionsf+numsignalsf):
            profilecheck += (profile[idx7007]*repmultipliersf[idx7007])
        if profilecheck != idx7004:
            print('Nathan you made a mistake')
    return sigurnsf, recurnsf, profiles_indexed, sigperdim, profilecaps

# function to populate the sender and receiver "urns" with random initial values
# also initializes an array that coorelates strategy profile with what the strategy is
@numba.jit(nopython=True, fastmath=False, parallel=False)
def replicatorexec_initialize(epsilonf, rngf, repmultipliersf, numprofilesf, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigurnsf, recurnsf, population_sizef):
    sigperdim = np.zeros((sigdimensionsf), dtype=np.int64)
    for idx7000 in range(0, sigdimensionsf):
        sigperdim[idx7000] = len(sigurnsf[idx7000][0])
    # adding in exec urns which will be sigdimensionsf long with a 0 if the agent does not attend to the dimension and 1 if the agent does
    execurns = np.zeros((population_sizef, sigdimensionsf), dtype=np.int64)
    for idx7001 in range(0, population_sizef):
        # generating signal urns to have all zeros, except a 1 in the location of the signal that is broadcasted
        for idx7002 in range(0, sigdimensionsf):
            csurn = sigurnsf[idx7002][idx7001]
#             cscumsum = np.cumsum(csurn)
#             csrand = (epsilonf+rngf.random())*cscumsum[-1]
            # for the null sig to not be in the initial strategy profiles map (0, 1] --> (1/(sigs in dim), 1]
            # can do this by multiplying csrand by (1-1/(sigs in dim)) and then adding 1/(sigs in dim)
#             csrand = 1/(len(cscumsum))+csrand*(1-(1/(len(cscumsum))))
            # end null sig modification # ***** removed for 54b implementation or maybe leave in
#             csdraw = np.zeros(len(cscumsum))
#             csdraw[cscumsum<csrand] = 1
            csdex = rngf.integers(0, len(csurn))
            cspopulate = np.zeros(len(csurn))
            cspopulate[csdex] = 1
            sigurnsf[idx7002][idx7001] = cspopulate
            # adding in executive urns random initialization
#             if epsilonf+rngf.random() < 0.5: # **** 54b removed **** removed this and following 6 lines for 55h
# #             if csdex != 0:
#                 execurns[idx7001][idx7002] = 1
#             else: # ***** added for 54b
#                 cspopulate[csdex] = 0
#                 cspopulate[0] = 1
#                 sigurnsf[idx7002][idx7001] = cspopulate
        # generating receiver urns similarly
        for idx7003 in range(0, numsignalsf):
#             acumsum = np.cumsum(recurnsf[idx7001][idx7003])
#             arand = (epsilonf+rngf.random())*acumsum[-1]
#             adraw = np.zeros(len(acumsum))
#             adraw[acumsum<arand] = 1
            adex = rngf.integers(0, numactionsf)
            recpopulate = np.zeros(numactionsf)
            recpopulate[adex] = 1
            recurnsf[idx7001][idx7003] = recpopulate
#     profiles_indexed = np.zeros((numprofilesf, sigdimensionsf+numsignalsf+sigdimensionsf), dtype=np.int64)
#     profile = np.zeros((sigdimensionsf+numsignalsf+sigdimensionsf), dtype=np.int64)
#     profilecaps = np.zeros((sigdimensionsf+numsignalsf+sigdimensionsf), dtype=np.int64)
#     for pcdex in range(0, sigdimensionsf+numsignalsf+sigdimensionsf): # ***** removed exec extension for 54b
    profiles_indexed = np.zeros((numprofilesf, sigdimensionsf+numsignalsf), dtype=np.int64)
    profile = np.zeros((sigdimensionsf+numsignalsf), dtype=np.int64)
    profilecaps = np.zeros((sigdimensionsf+numsignalsf), dtype=np.int64)
    for pcdex in range(0, sigdimensionsf+numsignalsf):
        if pcdex < sigdimensionsf:
            profilecaps[pcdex] = sigperdim[pcdex]
        elif pcdex < sigdimensionsf+numsignalsf:
            profilecaps[pcdex] = numactionsf
#         else:
#             profilecaps[pcdex] = 2 # executives only ever have two options for each dimension, dont attend 0 or 1 attend to the dimension
            
    for idx7004 in range(1, numprofilesf):
        profile[0] = (profile[0]+1)%profilecaps[0]
#         for idx7005 in range(1, sigdimensionsf+numsignalsf+sigdimensionsf): # **** 54b removed
        for idx7005 in range(1, sigdimensionsf+numsignalsf):
            zero = 0
            for idx7006 in range(0, idx7005):
                zero += profile[idx7006]
            if zero == 0:
                profile[idx7005] = (profile[idx7005]+1)%profilecaps[idx7005]
        profiles_indexed[idx7004] = profile
        # lets check that this was computed correctly
        profilecheck = 0
#         for idx7007 in range(0, sigdimensionsf+numsignalsf+sigdimensionsf): # **** 54b removed
        for idx7007 in range(0, sigdimensionsf+numsignalsf):
            profilecheck += (profile[idx7007]*repmultipliersf[idx7007])
        if profilecheck != idx7004:
            print('Nathan you made a mistake in profile indexing')
    return sigurnsf, recurnsf, execurns, profiles_indexed, sigperdim, profilecaps


# function to populate the sender and receiver "urns" with random initial values
# also initializes an array that coorelates strategy profile with what the strategy is
@numba.jit(nopython=True, fastmath=False, parallel=False)
def RothErevExec_initialize(epsilonf, rngf, repmultipliersf, numprofilesf, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigurnsf, recurnsf, population_sizef):
    sigperdim = np.zeros((sigdimensionsf), dtype=np.int64)
    for idx7000 in range(0, sigdimensionsf):
        sigperdim[idx7000] = len(sigurnsf[idx7000][0])
    # adding in exec urns which will be sigdimensionsf long with a 0 if the agent does not attend to the dimension and 1 if the agent does
    
    profile = np.zeros((sigdimensionsf+numsignalsf), dtype=np.int64)
    profilecaps = np.zeros((sigdimensionsf+numsignalsf), dtype=np.int64)
    for pcdex in range(0, sigdimensionsf+numsignalsf):
        if pcdex < sigdimensionsf:
            profilecaps[pcdex] = sigperdim[pcdex]
        elif pcdex < sigdimensionsf+numsignalsf:
            profilecaps[pcdex] = numactionsf
    return sigperdim, profilecaps



# need a function to get the mutual information based on Shannon entropy
@numba.jit(nopython=True, fastmath=False, parallel=False)
def mutual_info(numsignalsf, sigdimensionsf, sigmultipliersf, numactionsf, sigurnsf, recurnsf, population_sizef):
    sigperdim = np.zeros((sigdimensionsf), dtype=np.int64)
    for idx7000 in range(0, sigdimensionsf):
        sigperdim[idx7000] = len(sigurnsf[idx7000][0])
    signals_indexed =  np.zeros((numsignalsf, sigdimensionsf), dtype=np.int64)
    profile = np.zeros((sigdimensionsf), dtype=np.int64)
    for idx7004 in range(1, numsignalsf):
        profile[0] = (profile[0]+1)%sigperdim[0]
        for idx7005 in range(1, sigdimensionsf):
            zero = 0
            for idx7006 in range(0, idx7005):
                zero += profile[idx7006]
            if zero == 0:
                profile[idx7005] = (profile[idx7005]+1)%sigperdim[idx7005]
        signals_indexed[idx7004] = profile
    # okay now (i) get the probability of a signal being transmitted
    signal_probabilities = np.zeros((numsignalsf))
    for idx1400 in range(0, numsignalsf):
        sigprobcount = 0
        sigexpanded = signals_indexed[idx1400]
        for idx1401 in range(0, population_sizef):
            sigmul = 1
            for idx1402 in range(0, sigdimensionsf):
                sigmuldex = sigexpanded[idx1402]
                sigmul *= (sigurnsf[idx1402][idx1401][sigmuldex]/np.sum(sigurnsf[idx1402][idx1401]))
            sigprobcount += sigmul
        signal_probabilities[idx1400] = sigprobcount/population_sizef
    if np.absolute(1-np.sum(signal_probabilities)) > 0.01:
        print('signal probabilities dont equal 1')
        print(np.sum(signal_probabilities))
    # (ii) get the condtional probabilities of an action given a signal
    joint_probabilities = np.zeros((numactionsf, numsignalsf))
    jointcheck = 0
    for idx1403 in range(0, numactionsf):
        for idx1404 in range(0, numsignalsf):
            jointcount = 0
            for idx1405 in range(0, population_sizef):
                jointcount += (recurnsf[idx1405][idx1404][idx1403]/np.sum(recurnsf[idx1405][idx1404]))
            joint_probabilities[idx1403][idx1404] = ((jointcount/population_sizef)*signal_probabilities[idx1404])
            jointcheck += ((jointcount/population_sizef)*signal_probabilities[idx1404])
    if np.absolute(1-jointcheck) > 0.01:
        print(f'jointcheck is {jointcheck} not 1')
    # (iii) calculate the marginal probability of an action as a sum of the probabity of an action given a signal times the probability of that signal for all signals.
    action_probabilities = np.zeros((numactionsf))
    actcheck = 0
    for idx1406 in range(0, numactionsf):
        actcount = 0
        for idx1407 in range(0, numsignalsf):
            actcount += joint_probabilities[idx1406][idx1407]
        action_probabilities[idx1406] = actcount
        actcheck += actcount
    if np.absolute(1-actcheck) > 0.01:
        print(f'actcheck is {actcheck}  not 1')
    # (iv) finally, calculate mutual information given those three functions
    mutuinfo = 0
    for idx1408 in range(0, numactionsf):
        for idx1409 in range(0, numsignalsf):
            mutuinfodivisor = (action_probabilities[idx1408]*signal_probabilities[idx1409])
#             print(f'idx1408{idx1408}_idx1409{idx1409}')
#             print(idx1408, idx1409)
#             print(joint_probabilities[idx1408][idx1409])
#             print(mutuinfodivisor)
            if mutuinfodivisor == 0:
                mutuinfodivisor = 1/(10**10)
            mutuinfonumerator = joint_probabilities[idx1408][idx1409]
            if mutuinfonumerator != 0:
                mutuinfo += (mutuinfonumerator*np.log2(mutuinfonumerator/mutuinfodivisor))
#             print('mutuinfo')
#             print(mutuinfo)
    return mutuinfo

# @numba.jit(nopython=True, fastmath=False, parallel=False)
def average_mutual_info(numsignalsf, sigdimensionsf, sigmultipliersf, numactionsf, sigurnsf, recurnsf, population_sizef, runsf):
    averagemutuinfo = 0
    for runsdex in range(0, runsf):
        sigurnstyped = numba.typed.List(sigurnsf[runsdex])
        averagemutuinfo += mutual_info(numsignalsf, sigdimensionsf, sigmultipliersf, numactionsf, sigurnstyped, recurnsf[runsdex], population_sizef)
    averagemutuinfo = averagemutuinfo/runsf
    return averagemutuinfo



# calculating utiltiy under replicator dynamics of each profile for each type given the prevelence of the profiles
@numba.jit(nopython=True, fastmath=False, parallel=False)
def utilcompute(assort_multipliers, numtypesf, sigmultipliersf, sigdimensionsf, numprofilesf, profiles_indexed, profile_count_untyped, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf):
    profile_utility_bytype = np.zeros((numtypesf, numprofilesf))
    for idx1200 in range(0, numprofilesf):
        if profile_count_untyped[idx1200] != 0:
            Aprofile = profiles_indexed[idx1200]
            Asignal = 0
            for idx1200b in range(0, sigdimensionsf):
                Asignal += (Aprofile[idx1200b]*sigmultipliersf[idx1200b])
            for idx1201 in range(0, numprofilesf):
                if profile_count_untyped[idx1201] != 0:
                    Bprofile = profiles_indexed[idx1201]
                    Bsignal = 0
                    for idx1201b in range(0, sigdimensionsf):
                        Bsignal += (Bprofile[idx1201b]*sigmultipliersf[idx1201b])
                    Aaction = Aprofile[floor(sigdimensionsf+Bsignal)]
                    Baction = Bprofile[floor(sigdimensionsf+Asignal)]
                    hfmultiplier = assort_multipliers[floor(Asignal)][floor(Bsignal)]
                    if Aaction == Baction:
                        for idx1202 in range(0, numtypesf):
                            if idx1200 != idx1201:
                                profile_utility_bytype[idx1202][idx1200] += hfmultiplier*(profile_count_untyped[idx1201]*coordination_preferencesf[idx1202][Aaction])
                            else:
                                profile_utility_bytype[idx1202][idx1200] += hfmultiplier*((profile_count_untyped[idx1201]-1)*coordination_preferencesf[idx1202][Aaction])
                    else: # punish failures
                        for idx1202 in range(0, numtypesf):
                            if idx1200 != idx1201:
                                profile_utility_bytype[idx1202][idx1200] += hfmultiplier*(profile_count_untyped[idx1201]*genBSpunishf)
                            else:
                                profile_utility_bytype[idx1202][idx1200] += hfmultiplier*((profile_count_untyped[idx1201]-1)*genBSpunishf)
    
    return profile_utility_bytype

# use this funtion inside executilcompute to get all possible signals given some dimensions may be ignored
# the general shape of the second half of this code followed prior code from generating profiles indexed
@numba.jit(nopython=True, fastmath=False, parallel=False)
def expandedsignals(Aprofile, numsignalsf, sigdimensionsf, sigperdimf):
    # need to get executive profile and the collect all possible A sgnals if some dimension is not attended to
    Aexec = np.zeros((sigdimensionsf), dtype = np.int64)
    Asigbase = np.zeros((sigdimensionsf), dtype = np.int64)
    numAsignals = 1
    numAfreedims = 0
    for idx1200c in range(0, sigdimensionsf):
        Asigbase[idx1200c] = Aprofile[idx1200c]
        correctdex3 = sigdimensionsf+numsignalsf+idx1200c
        Aexec[idx1200c] = Aprofile[correctdex3]
        if Aprofile[correctdex3] == 0:
            numAsignals *= sigperdimf[idx1200c]
            numAfreedims += 1
    numAsignals = floor(numAsignals)
    numAattendeddims = sigdimensionsf-numAfreedims
    Asignalsexpanded = np.zeros((numAsignals, sigdimensionsf))
    Asignalsexpanded[0] = Asigbase.copy()
    if numAsignals != 1:
        Asigfluid = Aexec*Asigbase.copy()
        Asignalsexpanded[0] = Asigfluid.copy()
        Afreedims  = np.zeros((numAfreedims), dtype=np.int64)
        Afreedimscount = 0
        for idx1200d in range(0, sigdimensionsf):
            if Aexec[idx1200d] == 0:
                Afreedims[Afreedimscount] = idx1200d
                Afreedimscount += 1
        # now we have the specific dimensions that are free
        zerothdex = Afreedims[0]
        for idx1200e in range(1, numAsignals):
            Asigfluid[zerothdex] = (Asigfluid[zerothdex]+1)%sigperdimf[zerothdex]
            if numAfreedims != 1:
                for idx1200f in range(1, numAfreedims):
                    zero = 0
                    for idx1200g in range(0, idx1200f):
                        correctdimdex = Afreedims[idx1200g]
                        zero += Asigfluid[correctdimdex]
                    if zero == 0:
                        correctdimdex = Afreedims[idx1200f]
                        Asigfluid[correctdimdex] = (Asigfluid[correctdimdex]+1)%sigperdimf[correctdimdex]
            Asignalsexpanded[idx1200e] = Asigfluid
    return Asignalsexpanded, numAsignals, numAfreedims, numAattendeddims, Aexec



# calculating utiltiy under replicator dynamics of each profile for each type given the prevelence of the profiles
# executives version
@numba.jit(nopython=True, fastmath=False, parallel=False)
def executilcompute(assort_multipliers, sigcostf, sigperdimf, numtypesf, sigmultipliersf, sigdimensionsf, numsignalsf, numprofilesf, profiles_indexed, profile_count_untyped, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf):
    profile_utility_bytype = np.zeros((numtypesf, numprofilesf), dtype=np.float64)
    for idx1200 in range(0, numprofilesf):
        if profile_count_untyped[idx1200] != 0:
            Aprofile = profiles_indexed[idx1200]
            # adding in assortment signals need to calculate before zeroing out
            assortsigA = 0
            for idx1212asa in range(0, sigdimensionsf):
                assortsigA += (Aprofile[idx1212asa]*sigmultipliersf[idx1212asa])
            # end assort signals change
            totalAsigcost = 0
            for idx1212b in range(0, sigdimensionsf):
                if Aprofile[idx1212b] != 0:
                    totalAsigcost += 1
            totalAsigcost = totalAsigcost*sigcostf
            for idx1201 in range(0, numprofilesf):
                if profile_count_untyped[idx1201] != 0:
                    Bprofile = (profiles_indexed[idx1201]).copy()
                    # adding in assortment signals need to calculate before zeroing out
                    assortsigB = 0
                    for idx1212as in range(0, sigdimensionsf):
                        assortsigB += (Bprofile[idx1212as]*sigmultipliersf[idx1212as])
                    # end assort signals change
                    Aprofile_copy = Aprofile.copy()
                    for idx1212 in range(0, sigdimensionsf):
                        if Aprofile_copy[idx1212] == 0 or Bprofile[idx1212] == 0:
                            Aprofile_copy[idx1212] = 0
                            Bprofile[idx1212] = 0
                    Asignal = 0
                    for idx1200b in range(0, sigdimensionsf):
                        Asignal += (Aprofile_copy[idx1200b]*sigmultipliersf[idx1200b])
                    Bsignal = 0
                    for idx1201b in range(0, sigdimensionsf):
                        Bsignal += (Bprofile[idx1201b]*sigmultipliersf[idx1201b])
                    Aaction = Aprofile[floor(sigdimensionsf+Bsignal)]
                    Baction = Bprofile[floor(sigdimensionsf+Asignal)]
                    
                    hfmultiplier = assort_multipliers[floor(assortsigA)][floor(assortsigB)]
                    if Aaction == Baction:
                        for idx1202 in range(0, numtypesf):
                            if idx1200 != idx1201:
                                profile_utility_bytype[idx1202][idx1200] += hfmultiplier*(profile_count_untyped[idx1201]*(coordination_preferencesf[idx1202][Aaction]+totalAsigcost))
                            else:
                                profile_utility_bytype[idx1202][idx1200] += hfmultiplier*((profile_count_untyped[idx1201]-1)*(coordination_preferencesf[idx1202][Aaction]+totalAsigcost))
                    else: # punish failures
                        for idx1202 in range(0, numtypesf):
                            if idx1200 != idx1201:
                                profile_utility_bytype[idx1202][idx1200] += hfmultiplier*(profile_count_untyped[idx1201]*(genBSpunishf+totalAsigcost))
                            else:
                                profile_utility_bytype[idx1202][idx1200] += hfmultiplier*((profile_count_untyped[idx1201]-1)*(genBSpunishf+totalAsigcost))
    
    return profile_utility_bytype

# updating urns based on payoffs and costs under Roth-Erev dynamics
@numba.jit(nopython=True, fastmath=False, parallel=False)
def executilcomputeRE(agentprofilepicks, assort_multipliers, sigcostf, sigperdimf, sigmultipliersf, sigdimensionsf, numsignalsf, numactionsf, population_sizef, poptf, sigurnsf, recurnsf, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf):
    for idx1200 in range(0, population_sizef):
        Aprofile = (agentprofilepicks[idx1200]).copy()
        # adding in assortment signals need to calculate before zeroing out
        assortsigA = 0
        for idx1212asa in range(0, sigdimensionsf):
            assortsigA += (Aprofile[idx1212asa]*sigmultipliersf[idx1212asa])
        # end assort signals change
        totalAsigcost = 0
        for idx1212b in range(0, sigdimensionsf):
            if Aprofile[idx1212b] != 0:
                totalAsigcost += 1
        totalAsigcost = totalAsigcost*sigcostf
        Atype = poptf[idx1200]
        for idx1201mod in range(1, (population_sizef-idx1200)): # if we reinforce both A profile and B, then do not need to repeat 
                                                                # interactions. Hence population size - idx1200
            idx1201 = (idx1200+idx1201mod)%population_sizef # this ensures that agent doesnt play against theirself
            Bprofile = (agentprofilepicks[idx1201]).copy()
            # adding in assortment signals need to calculate before zeroing out
            assortsigB = 0
            for idx1212as in range(0, sigdimensionsf):
                assortsigB += (Bprofile[idx1212as]*sigmultipliersf[idx1212as])
            # end assort signals change
            totalBsigcost = 0
            Btype = poptf[idx1201]
            for idx1212b2 in range(0, sigdimensionsf):
                if Bprofile[idx1212b2] != 0:
                    totalBsigcost += 1
            totalBsigcost = totalBsigcost*sigcostf
            Aprofile_copy = Aprofile.copy()
            Bprofile_copy = Bprofile.copy()
            for idx1212 in range(0, sigdimensionsf):
                if Aprofile_copy[idx1212] == 0 or Bprofile[idx1212] == 0:
                    Aprofile_copy[idx1212] = 0
                    Bprofile_copy[idx1212] = 0
            Asignal = 0 # as seen by B
            for idx1200b in range(0, sigdimensionsf):
                Asignal += (Aprofile_copy[idx1200b]*sigmultipliersf[idx1200b])
            Bsignal = 0 # as seen by A
            for idx1201b in range(0, sigdimensionsf):
                Bsignal += (Bprofile_copy[idx1201b]*sigmultipliersf[idx1201b])
            Aaction = Aprofile[floor(sigdimensionsf+Bsignal)]
            Baction = Bprofile[floor(sigdimensionsf+Asignal)]
            
            Ahfmultiplier = assort_multipliers[floor(assortsigA)][floor(assortsigB)]
            Bhfmultiplier = assort_multipliers[floor(assortsigB)][floor(assortsigA)]
            if Aaction == Baction:
                Areinforcement = Ahfmultiplier*(coordination_preferencesf[Atype][Aaction]+totalAsigcost)
                Breinforcement = Bhfmultiplier*(coordination_preferencesf[Btype][Baction]+totalBsigcost)
            else:
                Areinforcement = Ahfmultiplier*(genBSpunishf+totalAsigcost)
                Breinforcement = Bhfmultiplier*(genBSpunishf+totalBsigcost)
            # now need to do the actual reinforcement since we can efficiently do both calculation and reinforecemnt in single step
            for idx1203 in range(0, sigdimensionsf):
                dimensionAsig = Aprofile[idx1203]
                dimensionBsig = Bprofile[idx1203]
                sigurnsf[idx1203][idx1200][dimensionAsig] += Areinforcement
                sigurnsf[idx1203][idx1201][dimensionBsig] += Breinforcement
            recurnsf[idx1200][Bsignal][Aaction] += Areinforcement
            recurnsf[idx1201][Asignal][Baction] += Breinforcement
    for idx1204 in range(0, sigdimensionsf):
        signals_in_this_dimension = sigperdimf[idx1204]
        for idx1205 in range(0, population_sizef):
            for idx1206 in range(0, signals_in_this_dimension):
                if sigurnsf[idx1204][idx1205][idx1206] < 1:
                    sigurnsf[idx1204][idx1205][idx1206] = 1
    for idx1207 in range(0, population_sizef):
        for idx1208 in range(0, numsignalsf):
            for idx1209 in range(0, numactionsf):
                if recurnsf[idx1207][idx1208][idx1209] < 1:
                    recurnsf[idx1207][idx1208][idx1209] = 1
    return sigurnsf, recurnsf


# updating urns based on payoffs and costs under Roth-Erev dynamics
@numba.jit(nopython=True, fastmath=False, parallel=False)
def executilcomputeREforget(forgetf, agentprofilepicks, assort_multipliers, sigcostf, sigperdimf, sigmultipliersf, sigdimensionsf, numsignalsf, numactionsf, population_sizef, poptf, sigurnsf, recurnsf, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf):
    for idx1204 in range(0, sigdimensionsf):
        signals_in_this_dimension = sigperdimf[idx1204]
        for idx1205 in range(0, population_sizef):
            for idx1206 in range(0, signals_in_this_dimension):
                sigurnsf[idx1204][idx1205][idx1206] *= forgetf
    for idx1207 in range(0, population_sizef):
        for idx1208 in range(0, numsignalsf):
            for idx1209 in range(0, numactionsf):
                recurnsf[idx1207][idx1208][idx1209] *= forgetf
    for idx1200 in range(0, population_sizef):
        Aprofile = (agentprofilepicks[idx1200]).copy()
        # adding in assortment signals need to calculate before zeroing out
        assortsigA = 0
        for idx1212asa in range(0, sigdimensionsf):
            assortsigA += (Aprofile[idx1212asa]*sigmultipliersf[idx1212asa])
        # end assort signals change
        totalAsigcost = 0
        for idx1212b in range(0, sigdimensionsf):
            if Aprofile[idx1212b] != 0:
                totalAsigcost += 1
        totalAsigcost = totalAsigcost*sigcostf
        Atype = poptf[idx1200]
        for idx1201mod in range(1, (population_sizef-idx1200)): # if we reinforce both A profile and B, then do not need to repeat 
                                                                # interactions. Hence population size - idx1200
            idx1201 = (idx1200+idx1201mod)%population_sizef # this ensures that agent doesnt play against theirself
            Bprofile = (agentprofilepicks[idx1201]).copy()
            # adding in assortment signals need to calculate before zeroing out
            assortsigB = 0
            for idx1212as in range(0, sigdimensionsf):
                assortsigB += (Bprofile[idx1212as]*sigmultipliersf[idx1212as])
            # end assort signals change
            totalBsigcost = 0
            Btype = poptf[idx1201]
            for idx1212b2 in range(0, sigdimensionsf):
                if Bprofile[idx1212b2] != 0:
                    totalBsigcost += 1
            totalBsigcost = totalBsigcost*sigcostf
            Aprofile_copy = Aprofile.copy()
            Bprofile_copy = Bprofile.copy()
            for idx1212 in range(0, sigdimensionsf):
                if Aprofile_copy[idx1212] == 0 or Bprofile[idx1212] == 0:
                    Aprofile_copy[idx1212] = 0
                    Bprofile_copy[idx1212] = 0
            Asignal = 0 # as seen by B
            for idx1200b in range(0, sigdimensionsf):
                Asignal += (Aprofile_copy[idx1200b]*sigmultipliersf[idx1200b])
            Bsignal = 0 # as seen by A
            for idx1201b in range(0, sigdimensionsf):
                Bsignal += (Bprofile_copy[idx1201b]*sigmultipliersf[idx1201b])
            Aaction = Aprofile[floor(sigdimensionsf+Bsignal)]
            Baction = Bprofile[floor(sigdimensionsf+Asignal)]
            
            Ahfmultiplier = assort_multipliers[floor(assortsigA)][floor(assortsigB)]
            Bhfmultiplier = assort_multipliers[floor(assortsigB)][floor(assortsigA)]
            if Aaction == Baction:
                Areinforcement = Ahfmultiplier*(coordination_preferencesf[Atype][Aaction]+totalAsigcost)
                Breinforcement = Bhfmultiplier*(coordination_preferencesf[Btype][Baction]+totalBsigcost)
            else:
                Areinforcement = Ahfmultiplier*(genBSpunishf+totalAsigcost)
                Breinforcement = Bhfmultiplier*(genBSpunishf+totalBsigcost)
            # now need to do the actual reinforcement since we can efficiently do both calculation and reinforecemnt in single step
            for idx1203 in range(0, sigdimensionsf):
                dimensionAsig = Aprofile[idx1203]
                dimensionBsig = Bprofile[idx1203]
                sigurnsf[idx1203][idx1200][dimensionAsig] += Areinforcement
                sigurnsf[idx1203][idx1201][dimensionBsig] += Breinforcement
            recurnsf[idx1200][Bsignal][Aaction] += Areinforcement
            recurnsf[idx1201][Asignal][Baction] += Breinforcement
    for idx1204 in range(0, sigdimensionsf):
        signals_in_this_dimension = sigperdimf[idx1204]
        for idx1205 in range(0, population_sizef):
            for idx1206 in range(0, signals_in_this_dimension):
                if sigurnsf[idx1204][idx1205][idx1206] < 1:
                    sigurnsf[idx1204][idx1205][idx1206] = 1
    for idx1207 in range(0, population_sizef):
        for idx1208 in range(0, numsignalsf):
            for idx1209 in range(0, numactionsf):
                if recurnsf[idx1207][idx1208][idx1209] < 1:
                    recurnsf[idx1207][idx1208][idx1209] = 1
    return sigurnsf, recurnsf
#     for idx1200 in range(0, numprofilesf):
#         if profile_count_untyped[idx1200] != 0:
#             Aprofile = (profiles_indexed[idx1200]).copy()
#             # lets get the number of dimensions that A attends to
#             numAattendeddims = 0
#             for idx1200c in range(0, sigdimensionsf):
# #                 correctdex3 = sigdimensionsf+numsignalsf+idx1200c # ***** removed for 54b
#                 correctdex3 = idx1200c
# #                 if Aprofile[correctdex3] == 1:  # ***** removed for 54b
#                 if Aprofile[correctdex3] != 0:
#                     numAattendeddims+=1
# #                 else: # lets go ahead and chande the A signal to zero if the dimension is not attended to  # ***** removed for 54b, Aprofile is already 0
# #                     Aprofile[idx1200c] = 0
#             totalAsigcost = numAattendeddims*sigcostf
#             for idx1201 in range(0, numprofilesf):
#                 if profile_count_untyped[idx1201] != 0:
#                     Aprofile_copy = Aprofile.copy()
#                     Bprofile = (profiles_indexed[idx1201]).copy()
#                     for idx1201c in range(0, sigdimensionsf): 
# #                         correctdex4 = sigdimensionsf+numsignalsf+idx1201c # ***** removed for 54b
#                         correctdex4 = idx1201c
#                         if Bprofile[correctdex4] == 0:
# #                             Bprofile[idx1201c] = 0 # ***** removed for 54b, Aprofile is already 0
#                             Aprofile_copy[idx1201c] = 0
#                         if Aprofile[correctdex4] == 0:
#                             Bprofile[idx1201c] = 0
#                     Asignal = 0
#                     for idx1200b in range(0, sigdimensionsf):
#                         Asignal += (Aprofile_copy[idx1200b]*sigmultipliersf[idx1200b]) #important to do this here so A is treated as null signal if B does not attend to dimension
#                     Bsignal = 0
#                     for idx1201b in range(0, sigdimensionsf):
#                         Bsignal += (Bprofile[idx1201b]*sigmultipliersf[idx1201b])
#                     Aaction = Aprofile[floor(sigdimensionsf+Bsignal)]
#                     Baction = Bprofile[floor(sigdimensionsf+Asignal)]
#                     dimcount = 0
#                     for idx1203 in range(0, sigdimensionsf):
#                         if Aprofile[idx1203] == Bprofile[idx1203] and Aprofile[idx1203] != 0: # don't use the copy here b/c A still pays signal cost if attending to dimension even if B does not attend to it
#                             dimcount += 1
#                     hfmultiplier = (base_connection_weightsf[floor(dimcount)])**homophily_factorf
#                     if Aaction == Baction:
#                         for idx1202 in range(0, numtypesf):
#                             if idx1200 != idx1201:
#                                 profile_utility_bytype[idx1202][idx1200] += hfmultiplier*(profile_count_untyped[idx1201]*(coordination_preferencesf[idx1202][Aaction]+totalAsigcost))
#                             else:
#                                 profile_utility_bytype[idx1202][idx1200] += hfmultiplier*((profile_count_untyped[idx1201]-1)*(coordination_preferencesf[idx1202][Aaction]+totalAsigcost))
#                     else: # punish failures
#                         for idx1202 in range(0, numtypesf):
#                             if idx1200 != idx1201:
#                                 profile_utility_bytype[idx1202][idx1200] += hfmultiplier*(profile_count_untyped[idx1201]*(genBSpunishf+totalAsigcost))
#                             else:
#                                 profile_utility_bytype[idx1202][idx1200] += hfmultiplier*((profile_count_untyped[idx1201]-1)*(genBSpunishf+totalAsigcost))

    
#     return profile_utility_bytype

# eliminating the cascade rounding fixed the statistical anomoly, but allowed a 1 in 10,000 chance of getting a zero error
# this function corrects the zero error when one occurs by assigning random strategy profiles to the given type
@numba.jit(nopython=True, fastmath=False, parallel=False)
def flip7zero_correction(slice_type, profile_count_typed_slice, numtypesf, rngf, population_sizef, numprofilesf, poptf):
    new_profile_count_typed_slice = (profile_count_typed_slice.copy())*0
    kprofile_list = rngf.integers(0, numprofilesf, size=population_sizef)
    for idx1100b in range(0, population_sizef):
        tdexb = poptf[idx1100b]
        if tdexb == slice_type:
            kprofile = kprofile_list[idx1100b]
            new_profile_count_typed_slice[kprofile] += 1

    return new_profile_count_typed_slice

# lets code the deterministic replicator dynamics
@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBSreplicator_single_tstep(numprofilesf, numtypesf, typestotal, profile_count_typed, agentprofiles, profile_utility_bytype, rngf, population_sizef, poptf):
    profile_count_untyped = np.zeros((numprofilesf), dtype=np.float64)
    # first average the utilities by type; I dont think the replicator dynamics weights this average by the number of individuals with that strategy b/c we're about to do that part
    # IMPORTANT need to get rid of utilities for profiles that do not exsit in the type before taking the average
    typesaverage = np.zeros((numtypesf), dtype=np.float64)
    for idx1300 in range(0, numtypesf):
        typeaveragecount = 0
        for idx1300b in range(0, numprofilesf):
            if profile_count_typed[idx1300][idx1300b] == 0:
                profile_utility_bytype[idx1300][idx1300b] = 0
            else:
                typeaveragecount += 1
        typesaverage[idx1300] = (np.sum(profile_utility_bytype[idx1300]))/typeaveragecount
#         if np.nanmean(profile_utility_bytype[idx1300]) == 0:
#             print('Nate is great!')
    # second get profile count typed and += each component by its value*(its utility - average utility for that type)
    for idx1301 in range(0, numtypesf):
        for idx1302 in range(0, numprofilesf):
            if profile_count_typed[idx1301][idx1302] != 0:
                profile_count_typed[idx1301][idx1302] += profile_count_typed[idx1301][idx1302]*(profile_utility_bytype[idx1301][idx1302]-typesaverage[idx1301])
                
                if profile_count_typed[idx1301][idx1302] < 0:
                    profile_count_typed[idx1301][idx1302] = 0
    # third, normalize and then multiply each value by the total number of agents of that type
    multipliers = np.zeros((numtypesf))
    for idx1303 in range(0, numtypesf):
        if np.sum(profile_count_typed[idx1303]) == 0:
            print(f'Nathan you made a 0 mistake for type {idx1303} a correction was made but if this happens')
            print('on more than a small handfull of runs, it should be investigated further')
            profile_count_typed[idx1303] = flip7zero_correction(idx1303, profile_count_typed[idx1303], numtypesf, rngf, population_sizef, numprofilesf, poptf)
            
        profile_count_typed[idx1303] = profile_count_typed[idx1303]*(typestotal[idx1303]/np.sum(profile_count_typed[idx1303]))
    # fourth, do the cascade rounding to get integer values for the number of agents with each strategy
    agentcount = 0
    for idx1304 in range(0, numtypesf):
        floattotal = 0
        integertotal = 0
        for idx1305 in range(0, numprofilesf):
            floattotal += profile_count_typed[idx1304][idx1305]
            intround = np.around(floattotal)
            incriment = intround-integertotal
            # profile_count_typed[idx1304][idx1305] = incriment # <--- this caused a statistical anomoly that I can't explain
            # profile_count_untyped[idx1305] += incriment # <--- this caused a statistical anomoly that I can't explain
            profile_count_untyped[idx1305] += profile_count_typed[idx1304][idx1305]
            integertotal += incriment
            if incriment != 0:
                agentcountplus = agentcount + incriment
                for idx1306 in range(agentcount, agentcountplus):
                    agentprofiles[idx1306] = idx1305
                agentcount = agentcountplus
    
    return profile_count_typed, profile_count_untyped, agentprofiles

@numba.jit(nopython=True, fastmath=False, parallel=False)
def numbagreater(axis0len, axis1len, nbarray, nbvalue):
    for idxNB0 in range(0, axis0len):
        for idxNB1 in range(0, axis1len):
            if nbarray[idxNB0][idxNB1] > nbvalue:
                nbarray[idxNB0][idxNB1] = 0
            else:
                nbarray[idxNB0][idxNB1] = 1
    return nbarray

@numba.jit(nopython=True, fastmath=False, parallel=False)
def muterandoms10(rngf, mutateratef, population_sizef, numprofilesf):
    mutations10 = rngf.random((10, population_sizef))
    mutations10 = numbagreater(10, population_sizef, mutations10, mutateratef)
    uniformsums = np.sum(mutations10, axis=1)
    unimax = np.max(uniformsums)
    newprofiles10 = rngf.integers(0, numprofilesf, (10, floor(unimax)))
    return mutations10, newprofiles10

# lets code the deterministic replicator dynamics
@numba.jit(nopython=True, fastmath=False, parallel=False)
def repmutation(numtypesf, numprofilesf, profile_count_typed, profile_count_untyped, mutations, newprofiles):
    profile_count_untyped = np.zeros((numprofilesf), dtype=np.float64)
    profile_count_typed_new = profile_count_typed.copy()
    countbounds = np.zeros((2), dtype=np.int64) # count lower and upper bounds
    countmutations = np.zeros((1), dtype=np.int64)
    for idx2000 in range(0, numtypesf):
        for idx2001 in range(0, numprofilesf):
            if profile_count_typed[idx2000][idx2001] != 0:
                profile_count_untyped[idx2001] += profile_count_typed[idx2000][idx2001]
                countbounds[1] += profile_count_typed[idx2000][idx2001]
#                 if countbounds[1] > 20000: # just using this line for troubleshooting
#                     print(f'at idx2000 = {idx2000} and idx2001 = {idx2001}')
#                     print(f'profilecount typed sum = {np.sum(profile_count_typed)}')
#                     print(f' count bounds is = {countbounds}')
                for idx2002 in range(countbounds[0], countbounds[1]):
                    if mutations[idx2002] == 1:
                        profile_count_untyped[idx2001] -= 1
                        profile_count_typed_new[idx2000][idx2001] -= 1
                        mutatations_index = countmutations[0]
                        mutation_profile = newprofiles[mutatations_index]
                        profile_count_untyped[mutation_profile] += 1
                        profile_count_typed_new[idx2000][mutation_profile] += 1
                        countmutations[0] += 1
                countbounds[0] = countbounds[1]
    
    return profile_count_typed_new, profile_count_untyped

@numba.jit(nopython=True, fastmath=False, parallel=False)
def numbagreater3d(axis0len, axis1len, axis2len, nbarray, nbvalue):
    for idxNB0 in range(0, axis0len):
        for idxNB1 in range(0, axis1len):
            for idxNB2 in range(0, axis2len):
                if nbarray[idxNB0][idxNB1][idxNB2] > nbvalue:
                    nbarray[idxNB0][idxNB1][idxNB2] = 0
                else:
                    nbarray[idxNB0][idxNB1][idxNB2] = 1
    return nbarray

@numba.jit(nopython=True, fastmath=False, parallel=False)
def muterandoms10_smart(rngf, mutateratef, population_sizef, numprofilesf, stringlength):
    mutations10 = rngf.random((10, population_sizef))
    mutations10 = numbagreater(10, population_sizef, mutations10, mutateratef[0])
    uniformsums = np.sum(mutations10, axis=1)
    unimax = np.max(uniformsums)
    newprofiles10 = rngf.integers(0, numprofilesf, (10, floor(unimax)))
    muteprofilestrings10 = rngf.random((10, floor(unimax), stringlength))
    muteprofilestrings10 = numbagreater3d(10, floor(unimax), stringlength, muteprofilestrings10, mutateratef[1])
    return mutations10, newprofiles10, muteprofilestrings10

# lets code the deterministic replicator dynamics
@numba.jit(nopython=True, fastmath=False, parallel=False)
def repmutation_smart(repmultipliersf, profilecaps, profiles_indexed, numtypesf, numprofilesf, profile_count_typed, profile_count_untyped, mutations, newprofiles, muteprofilestrings):
    profile_count_untyped = np.zeros((numprofilesf), dtype=np.float64)
    profile_count_typed_new = profile_count_typed.copy()
    countbounds = np.zeros((2), dtype=np.int64) # count lower and upper bounds
    countmutations = np.zeros((1), dtype=np.int64)
    profile_length = len(repmultipliersf)
    for idx2000 in range(0, numtypesf):
        for idx2001 in range(0, numprofilesf):
            if profile_count_typed[idx2000][idx2001] != 0:
                profile_count_untyped[idx2001] += profile_count_typed[idx2000][idx2001]
                countbounds[1] += profile_count_typed[idx2000][idx2001]
#                 if countbounds[1] > 20000: # just using this line for troubleshooting
#                     print(f'at idx2000 = {idx2000} and idx2001 = {idx2001}')
#                     print(f'profilecount typed sum = {np.sum(profile_count_typed)}')
#                     print(f' count bounds is = {countbounds}')
                for idx2002 in range(countbounds[0], countbounds[1]):
                    if mutations[idx2002] == 1:
                        profile_count_untyped[idx2001] -= 1
                        profile_count_typed_new[idx2000][idx2001] -= 1
                        if profile_count_typed_new[idx2000][idx2001] < 0:
                            profile_count_typed_new[idx2000][idx2001] = 0
                            if profile_count_untyped[idx2001] < 0:
                                profile_count_untyped[idx2001] = 0
                        mutatations_index = countmutations[0]
                        mutation_profile = newprofiles[mutatations_index]
                        mutation_profile_expanded = (profiles_indexed[mutation_profile]).copy()
                        mutation_profilestring = muteprofilestrings[mutatations_index]
#                         mutation_profile_expanded = np.multiply(mutation_profile_expanded, mutation_profilestring)
                        old_profile = (profiles_indexed[idx2001]).copy()
                        new_profile = 0
                        for idx2003 in range(0, profile_length):
                            if mutation_profilestring[idx2003] == 1:
                                new_profile += ((mutation_profile_expanded[idx2003])*repmultipliersf[idx2003])
                            else:
                                new_profile += ((old_profile[idx2003])*repmultipliersf[idx2003])
                        new_profile = floor(new_profile)
                        profile_count_untyped[new_profile] += 1
                        profile_count_typed_new[idx2000][new_profile] += 1
                        countmutations[0] += 1
                countbounds[0] = countbounds[1]
    
    return profile_count_typed_new, profile_count_untyped


# function to convert agent profiles back into sig and rec urns
@numba.jit(nopython=True, fastmath=False, parallel=False)
def repconvert(sigurnsf, recurnsf, agentprofiles, profiles_indexed, sigdimensionsf, numsignalsf, population_sizef):
    for idx1400 in range(0, population_sizef):
        currentdex = agentprofiles[idx1400]
        currentprofile = profiles_indexed[currentdex]
        for idx1401 in range(0, sigdimensionsf):
            sigurnsf[idx1401][idx1400] = 0
            sigdex = currentprofile[idx1401]
            sigurnsf[idx1401][idx1400][sigdex] = 1
        for idx1402 in range(0, numsignalsf):
            recurnsf[idx1400][idx1402] = 0
            profiledex = sigdimensionsf+idx1402
            recdex = currentprofile[profiledex]
            recurnsf[idx1400][idx1402][recdex] = 1
    return sigurnsf, recurnsf

# function to convert agent profiles back into sig and rec urns
@numba.jit(nopython=True, fastmath=False, parallel=False)
def execrepconvert(sigurnsf, recurnsf, execurnsf, agentprofiles, profiles_indexed, sigdimensionsf, numsignalsf, population_sizef):
    for idx1400 in range(0, population_sizef):
        currentdex = agentprofiles[idx1400]
        currentprofile = profiles_indexed[currentdex]
        for idx1401 in range(0, sigdimensionsf):
            sigurnsf[idx1401][idx1400] = 0
            sigdex = currentprofile[idx1401]
            sigurnsf[idx1401][idx1400][sigdex] = 1
        for idx1402 in range(0, numsignalsf):
            recurnsf[idx1400][idx1402] = 0
            profiledex = sigdimensionsf+idx1402
            recdex = currentprofile[profiledex]
            recurnsf[idx1400][idx1402][recdex] = 1
        for idx1403 in range(0, sigdimensionsf):
#             execprofiledex = sigdimensionsf+numsignalsf+idx1403 # removed for 54b
            execprofiledex = idx1403
            execvalue = currentprofile[execprofiledex]
            if execvalue == 0:
                execurnsf[idx1400][idx1403] = execvalue
            else:
                execurnsf[idx1400][idx1403] = 1
    return sigurnsf, recurnsf, execurnsf

# function to convert agent profiles back into sig and rec urns
@numba.jit(nopython=True, fastmath=False, parallel=False)
def execrepconvertRE(sigurnsf, recurnsf, execurnsf, sigdimensionsf, numsignalsf, population_sizef):
    sigurnsf_4 = sigurnsf.copy()
    recurnsf_4 = recurnsf.copy()
    for idx1400 in range(0, population_sizef):
        for idx1403 in range(0, sigdimensionsf):
            execvalue = np.argmax(sigurnsf_4[idx1403][idx1400])
            if execvalue == 0:
                execurnsf[idx1400][idx1403] = execvalue
            else:
                execurnsf[idx1400][idx1403] = 1
    return sigurnsf, recurnsf, execurnsf

# lets code the deterministic replicator dynamics
@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBSreplicator_first_tstep(repmultipliersf, genBSpunishf, numsignalsf, numprofilesf, numtypesf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, profiles_indexed):
    
    typestotal = np.zeros((numtypesf), dtype=np.int64)
    profile_count_typed = np.zeros((numtypesf, numprofilesf), dtype=np.float64)
    profile_count_untyped = np.zeros((numprofilesf), dtype=np.float64)
    agentprofiles = np.zeros((population_sizef), dtype=np.int64)
    for idx1100 in range(0, population_sizef):
        tdex = poptf[idx1100]
        typestotal[tdex] += 1
        # getting this agents strategy
        stratdex = 0
        for idx1101 in range(0, sigdimensionsf):
            stratdex += np.argmax(sigurnsf[idx1101][idx1100])*repmultipliersf[idx1101]
        for idx1102 in range(0, numsignalsf):
            correctdex = sigdimensionsf+idx1102
            stratdex += np.argmax(recurnsf[idx1100][idx1102])*repmultipliersf[correctdex]
        stratdex = floor(stratdex)
        profile_count_typed[tdex][stratdex] += 1
        profile_count_untyped[stratdex] += 1
        agentprofiles[idx1100] = stratdex
    # lets have a function that takes coord preferences and types total and returns the utility of each strategy profile present for each type
    profile_utility_bytype = utilcompute(numtypesf, sigmultipliersf, sigdimensionsf, numprofilesf, profiles_indexed, profile_count_untyped, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf)
    # now that we have the utilities by type 
    # first average the utilities by type; I dont think the replicator dynamics weights this average by the number of individuals with that strategy b/c we're about to do that part
    # second get profile count typed and += each component by its value*(its utility - average utility for that type)
    # third, normalize and then multiply each value by the total number of agents of that type
    # fourth, do the cascade rounding to get integer values for the number of agents with each strategy
    # should be able to doe this such that only need to return the new profile count and can wait to convert back to the sigurns and recurns 
    # format until the end of the simulation
    profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicator_single_tstep(numprofilesf, numtypesf, typestotal, profile_count_typed, agentprofiles, profile_utility_bytype)
    
    return typestotal, profile_count_typed, profile_count_untyped, agentprofiles

@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBSreplicatorexec_k_step(numtypesf, rngf, population_sizef, numprofilesf, poptf):
    typestotal = np.zeros((numtypesf), dtype=np.float64)
    profile_count_typed = np.zeros((numtypesf, numprofilesf), dtype=np.float64)
    profile_count_untyped = np.zeros((numprofilesf), dtype=np.float64)
    agentprofiles = np.zeros((population_sizef), dtype=np.int64)
    kprofile_list = rngf.integers(0, numprofilesf, size=population_sizef)
    for idx1100 in range(0, population_sizef):
        tdex = poptf[idx1100]
        typestotal[tdex] += 1
        kprofile = kprofile_list[idx1100]
        agentprofiles[idx1100] = kprofile
        profile_count_typed[tdex][kprofile] += 1
        profile_count_untyped[kprofile] += 1
        
    return typestotal, profile_count_typed, profile_count_untyped, agentprofiles

# lets code the deterministic replicator dynamics
@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBSreplicatorexec_first_tstep(sigcostf, sigperdimf, repmultipliersf, genBSpunishf, numsignalsf, numprofilesf, numtypesf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, execurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, profiles_indexed):
    
    typestotal = np.zeros((numtypesf), dtype=np.float64)
    profile_count_typed = np.zeros((numtypesf, numprofilesf), dtype=np.float64)
    profile_count_untyped = np.zeros((numprofilesf), dtype=np.float64)
    agentprofiles = np.zeros((population_sizef), dtype=np.int64)
    for idx1100 in range(0, population_sizef):
        tdex = poptf[idx1100]
        typestotal[tdex] += 1
        # getting this agents strategy
        stratdex = 0
        for idx1101 in range(0, sigdimensionsf):
            stratdex += np.argmax(sigurnsf[idx1101][idx1100])*repmultipliersf[idx1101]
        for idx1102 in range(0, numsignalsf):
            correctdex = sigdimensionsf+idx1102
            stratdex += np.argmax(recurnsf[idx1100][idx1102])*repmultipliersf[correctdex]
#         for idx1103 in range(0, sigdimensionsf): # ***** removed for 54b
#             correctdex2 = sigdimensionsf+numsignalsf+idx1103
#             stratdex += execurnsf[idx1100][idx1103]*repmultipliersf[correctdex2] # dont need to argmax because never used roth-erev style urns for replicator executives
        stratdex = floor(stratdex)
        profile_count_typed[tdex][stratdex] += 1
        profile_count_untyped[stratdex] += 1
        agentprofiles[idx1100] = stratdex
    # lets have a function that takes coord preferences and types total and returns the utility of each strategy profile present for each type
    profile_utility_bytype = executilcompute(sigcostf, sigperdimf, numtypesf, sigmultipliersf, sigdimensionsf, numsignalsf, numprofilesf, profiles_indexed, profile_count_untyped, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf)
    # now that we have the utilities by type 
    # first average the utilities by type; I dont think the replicator dynamics weights this average by the number of individuals with that strategy b/c we're about to do that part
    # second get profile count typed and += each component by its value*(its utility - average utility for that type)
    # third, normalize and then multiply each value by the total number of agents of that type
    # fourth, do the cascade rounding to get integer values for the number of agents with each strategy
    # should be able to doe this such that only need to return the new profile count and can wait to convert back to the sigurns and recurns 
    # format until the end of the simulation
    profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicator_single_tstep(numprofilesf, numtypesf, typestotal, profile_count_typed, agentprofiles, profile_utility_bytype)
    
    return typestotal, profile_count_typed, profile_count_untyped, agentprofiles
    
@numba.jit(nopython=True, fastmath=False, parallel=False)
def greetings_results(poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef):
    # B.i lets see (by agent type) an aggregate of each agent's most frequent signal
    socialsig_aggregate = np.zeros((numtypesf, numsignalsf), dtype=np.int64)
    action_aggregate = np.zeros((numtypesf, numsignalsf, numtypesf), dtype=np.int64)
    for idx9000 in range(0, population_sizef):
        cursig = 0
        for idx9001 in range(0, sigdimensionsf):
            cursig += np.argmax(sigurnsf[idx9001][idx9000])*sigmultipliersf[idx9001]
        curtype = poptf[idx9000]
        socialsig_aggregate[curtype][cursig] += 1
    # B.ii lets see (by agent type) for each signal what is the most frequent action
    for idx9002 in range(0, population_sizef):
        for idx9003 in range(0, numsignalsf):
            curaction = np.argmax(recurnsf[idx9002][idx9003])
            curtype = poptf[idx9002]
            action_aggregate[curtype][idx9003][curaction] += 1
    # B.iii let's further simplify the data by generalizing over most common behaviors of each type to report whether the
    # most likely action for a pairing is the correct greeting
    simple_stat_000 = np.zeros((numtypesf, numtypesf), dtype=np.int64)
    for idx9004 in range(0, numtypesf):
        for idx9005 in range(0, numtypesf):
            # 5's most likely social signal
            cursig = np.argmax(socialsig_aggregate[idx9005])
            # 4's most likely action given 5's most likely signal
            curaction = np.argmax(action_aggregate[idx9004][cursig])
            # success condition
            if curaction == idx9005: # note idx9005 is 5's type
                simple_stat_000[idx9004][idx9005] = 1
    return simple_stat_000, socialsig_aggregate, action_aggregate

@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBS_results(numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef):
    # B.i lets see (by agent type) an aggregate of each agent's most frequent signal
    socialsig_aggregate = np.zeros((numtypesf, numsignalsf), dtype=np.int64)
    action_aggregate = np.zeros((numtypesf, numsignalsf, numactionsf), dtype=np.int64)
    for idx9000 in range(0, population_sizef):
        cursig = 0
        for idx9001 in range(0, sigdimensionsf):
            cursig += np.argmax(sigurnsf[idx9001][idx9000])*sigmultipliersf[idx9001]
        curtype = poptf[idx9000]
        socialsig_aggregate[curtype][cursig] += 1
    # B.ii lets see (by agent type) for each signal what is the most frequent action
    for idx9002 in range(0, population_sizef):
        for idx9003 in range(0, numsignalsf):
            curaction = np.argmax(recurnsf[idx9002][idx9003])
            curtype = poptf[idx9002]
            action_aggregate[curtype][idx9003][curaction] += 1
    # B.iii let's further simplify the data by generalizing over most common behaviors of each type to report whether the
    # most likely action for a pairing is the correct greeting
    simple_stat_001 = np.zeros((numtypesf, numtypesf, numactionsf), dtype=np.int64)
    simple_stat_002 = np.zeros((numtypesf, numtypesf, numactionsf), dtype=np.int64)
    for idx9004 in range(0, numtypesf):
        for idx9005 in range(0, numtypesf):
            # 5's most likely social signal
            cursig = np.argmax(socialsig_aggregate[idx9005])
            # 4's most likely action given 5's most likely signal
            curaction = np.argmax(action_aggregate[idx9004][cursig])
            # record 4's most likely action given 5's signal
            simple_stat_001[idx9004][idx9005][curaction] = 1
            if socialsig_aggregate[idx9005][cursig] >  (np.sum(socialsig_aggregate[idx9005]))*.75 and action_aggregate[idx9004][cursig][curaction] > (np.sum(action_aggregate[idx9004][cursig]))*.75 :
                simple_stat_002[idx9004][idx9005][curaction] = 1
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate

# for roth erev we need the urns based measure but with executives accounted for
@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBS_resultsRE(sigperdimf, numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef):
    # B.i lets see (by agent type) an aggregate of each agent's most frequent signal
    socialsig_aggregate = np.zeros((numtypesf, numsignalsf), dtype=np.int64)
    maxdimsig = np.max(sigperdimf)
    socialsig_aggregate_expanded = np.zeros((numtypesf, sigdimensionsf, maxdimsig), dtype=np.int64)
    action_aggregate = np.zeros((numtypesf, numsignalsf, numactionsf), dtype=np.int64)
    for idx9000 in range(0, population_sizef):
        cursig = 0
        curtype = poptf[idx9000]
        for idx9001 in range(0, sigdimensionsf):
            cursig += np.argmax(sigurnsf[idx9001][idx9000])*sigmultipliersf[idx9001]
            socialsig_aggregate_expanded[curtype][idx9001][floor(np.argmax(sigurnsf[idx9001][idx9000]))]+=1
        socialsig_aggregate[curtype][cursig] += 1
    # B.ii lets see (by agent type) for each signal what is the most frequent action
    for idx9002 in range(0, population_sizef):
        for idx9003 in range(0, numsignalsf):
            curaction = np.argmax(recurnsf[idx9002][idx9003])
            curtype = poptf[idx9002]
            action_aggregate[curtype][idx9003][curaction] += 1
    # B.iii let's further simplify the data by generalizing over most common behaviors of each type to report whether the
    # most likely action for a pairing is the correct greeting
    simple_stat_001 = np.zeros((numtypesf, numtypesf, numactionsf), dtype=np.int64)
    simple_stat_002 = np.zeros((numtypesf, numtypesf, numactionsf), dtype=np.int64)
    for idx9004 in range(0, numtypesf):
        x4sig = np.zeros(sigdimensionsf, dtype=np.int64)
        for idx9004b in range(0, sigdimensionsf):
            x4sig[idx9004b] = np.argmax(socialsig_aggregate_expanded[idx9004][idx9004b])
        for idx9005 in range(0, numtypesf):
            # 5's most likely social signal AS SEEN BY 4 GIVEN ATTENTION
            cursig5 = 0
            for idx9005b in range(0, sigdimensionsf):
                if x4sig[idx9005b] !=0:
                    cursig5 += (np.argmax(socialsig_aggregate_expanded[idx9005][idx9005b]))*sigmultipliersf[idx9005b]
            cursig5 = floor(cursig5)
            # 4's most likely action given 5's most likely signal
            curaction = np.argmax(action_aggregate[idx9004][cursig5])
            # record 4's most likely action given 5's signal
            simple_stat_001[idx9004][idx9005][curaction] = 1
            if socialsig_aggregate[idx9005][cursig5] >  (np.sum(socialsig_aggregate[idx9005]))*.75 and action_aggregate[idx9004][cursig5][curaction] > (np.sum(action_aggregate[idx9004][cursig5]))*.75 :
                simple_stat_002[idx9004][idx9005][curaction] = 1
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate

@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBSexec_results(snapcount, sigperdimf, profiles_indexed, agentprofiles, numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, execurnsf, population_sizef):
    # B.i lets see (by agent type) an aggregate of each agent's most frequent signal
    socialsig_aggregate = np.zeros((numtypesf, numsignalsf), dtype=np.int64)
    action_aggregate = np.zeros((numtypesf, numsignalsf, numactionsf), dtype=np.float64)
    for idx9000 in range(0, population_sizef):
        agprofiledex = agentprofiles[idx9000]
        agprofile = (profiles_indexed[agprofiledex]).copy()
#         for idx9000c in range(0, sigdimensionsf): # ***** removed for 54b
#             correctdex5 = sigdimensionsf+numsignalsf+idx9000c
#             if agprofile[correctdex5] == 0:
#                 agprofile[idx9000c] = 0
        cursig = 0
        for idx9001 in range(0, sigdimensionsf):
            cursig += agprofile[idx9001]*sigmultipliersf[idx9001]
        curtype = poptf[idx9000]
        socialsig_aggregate[curtype][cursig] += 1
    # lets get signals_indexed analogous to profiles_indexed
    signals_indexed = np.zeros((numsignalsf, sigdimensionsf), dtype=np.int64)
    itersignal = np.zeros((sigdimensionsf), dtype=np.int64)
    for idx9104 in range(1, numsignalsf):
        itersignal[0] = (itersignal[0]+1)%sigperdimf[0]
        for idx9105 in range(1, sigdimensionsf):
            zero = 0
            for idx9106 in range(0, idx9105):
                zero += itersignal[idx9106]
            if zero == 0:
                itersignal[idx9105] = (itersignal[idx9105]+1)%sigperdimf[idx9105]
        signals_indexed[idx9104] = itersignal
    # B.ii lets see (by agent type) for each signal what is the most frequent action
    for idx9002 in range(0, population_sizef):
        # stoped here before rooftop **********************************************************************
        recprofiledex = agentprofiles[idx9002]
        recprofile = profiles_indexed[recprofiledex]
        curtype = poptf[idx9002]
        for idx9003 in range(0, numsignalsf):
            seensignal = (signals_indexed[idx9003]).copy()
            for idx9003c in range(0, sigdimensionsf):
#                 correctdex6 = sigdimensionsf+numsignalsf+idx9003c # ***** removed for 54b
                correctdex6 = idx9003c
                if recprofile[correctdex6] == 0:
                    seensignal[idx9003c] = 0
            recsignal = 0
            for idx9003d in range(0, sigdimensionsf):
                recsignal += seensignal[idx9003d]*sigmultipliersf[idx9003d]
            curactiondex = floor(recsignal+sigdimensionsf)
            curaction = recprofile[curactiondex]
            action_aggregate[curtype][idx9003][curaction] += 1
            
    # B.iii let's further simplify the data by generalizing over most common behaviors of each type to report whether the
    # most likely action for a pairing is the correct greeting
    simple_stat_001 = np.zeros((numtypesf, numtypesf, numactionsf), dtype=np.int64)
    simple_stat_002 = np.zeros((numtypesf, numtypesf, numactionsf), dtype=np.int64)
    for idx9004 in range(0, numtypesf):
        for idx9005 in range(0, numtypesf):
            # 5's most likely social signal
            cursig = np.argmax(socialsig_aggregate[idx9005])
            # 4's most likely action given 5's most likely signal
            curaction = np.argmax(action_aggregate[idx9004][cursig])
            # record 4's most likely action given 5's signal
            simple_stat_001[idx9004][idx9005][curaction] = 1
            if socialsig_aggregate[idx9005][cursig] >  (np.sum(socialsig_aggregate[idx9005]))*.75 and action_aggregate[idx9004][cursig][curaction] > (np.sum(action_aggregate[idx9004][cursig]))*.75 :
                simple_stat_002[idx9004][idx9005][curaction] = 1
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate



@numba.jit(nopython=True, fastmath=False, parallel=False)
def greetings_single_tstep(reinforcementf, punishmentf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf):
    # first each agent selects social signal
    sigdraws = social_draws(sigurnsf, population_sizef, sigdimensionsf, rngf, epsilonf)
    # second generate connection weights for assortment
    conweightsf = connections_init(population_sizef, sigdraws, base_connection_weightsf, homophily_factorf)
    # third generate pairings of agents
    randperm01f = rngf.permutation(population_sizef)
    pairsf = pairings(randperm01f, rngf, population_sizef, conweightsf, epsilonf)
    # fourth agents draw from their receiver urns to determine their action based on the social signal of their partner
    recdraws = receiver_draws(population_sizef, pairsf, rngf, recurnsf, sigdraws, sigmultipliersf, sigdimensionsf, epsilonf)
    # fifth agents actions are rewarded or punished based on whether they performed the correct action
    sigurnsf, recurnsf = greetings_check_success(sigdimensionsf, sigmultipliersf, reinforcementf, punishmentf, population_sizef, sigurnsf, pairsf, sigdraws, recurnsf, recdraws, poptf)
    # finally return the new sender and receiver urns
    return sigurnsf, recurnsf


@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBS_single_tstep(genBSpunishf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf):
    # first each agent selects social signal
    sigdraws = social_draws(sigurnsf, population_sizef, sigdimensionsf, rngf, epsilonf)
    # second generate connection weights for assortment
    conweightsf = connections_init(population_sizef, sigdraws, base_connection_weightsf, homophily_factorf)
    # third generate pairings of agents
    randperm01f = rngf.permutation(population_sizef)
    pairsf = pairings(randperm01f, rngf, population_sizef, conweightsf, epsilonf)
    # fourth agents draw from their receiver urns to determine their action based on the social signal of their partner
    recdraws = receiver_draws(population_sizef, pairsf, rngf, recurnsf, sigdraws, sigmultipliersf, sigdimensionsf, epsilonf)
    # fifth agents actions are rewarded or punished based on whether they performed the correct action ** adding the social learning into this step
    sigurnsf, recurnsf = genBS_check_success(genBSpunishf, sigdimensionsf, sigmultipliersf, numactionsf, coordination_preferencesf, population_sizef, sigurnsf, pairsf, sigdraws, recurnsf, recdraws, poptf, rngf)
    # finally return the new sender and receiver urns
    return sigurnsf, recurnsf


# function to create an [numsignals x numsignals] array that tells how many dimenstions in common the respective signals have
# regularand execs version have to be different. this is the one without execs
@numba.jit(nopython=True, fastmath=False, parallel=False)
def assort_array_create(homophily_factorf, sigperdimf, numsignalsf, sigdimensionsf, base_connection_weightsf):
    # getting signals indexed analogous to profiles indexed
    signals_indexed = np.zeros((numsignalsf, sigdimensionsf), dtype=np.int64)
    itersignal = np.zeros((sigdimensionsf), dtype=np.int64)
    for idx9104 in range(1, numsignalsf):
        itersignal[0] = (itersignal[0]+1)%sigperdimf[0]
        for idx9105 in range(1, sigdimensionsf):
            zero = 0
            for idx9106 in range(0, idx9105):
                zero += itersignal[idx9106]
            if zero == 0:
                itersignal[idx9105] = (itersignal[idx9105]+1)%sigperdimf[idx9105]
        signals_indexed[idx9104] = itersignal
        
    assort_array = np.zeros((numsignalsf, numsignalsf), dtype=np.int64)
    for idx9200 in range(0, numsignalsf):
        for idx9201 in range(0, numsignalsf):
            sigsamecount = 0
            for idx9202 in range(0, sigdimensionsf):
                if signals_indexed[idx9200][idx9202] == signals_indexed[idx9201][idx9202]:
                    sigsamecount += 1
            assort_array[idx9200][idx9201] = (base_connection_weightsf[sigsamecount])**homophily_factorf
    return assort_array

# function to create an [numsignals x numsignals] array that tells how many dimenstions in common the respective signals have
# regularand execs version have to be different. this is the one with execs
@numba.jit(nopython=True, fastmath=False, parallel=False)
def execassort_array_create(homophily_factorf, sigperdimf, numsignalsf, sigdimensionsf, base_connection_weightsf):
    # getting signals indexed analogous to profiles indexed
    signals_indexed = np.zeros((numsignalsf, sigdimensionsf), dtype=np.int64)
    itersignal = np.zeros((sigdimensionsf), dtype=np.int64)
    for idx9104 in range(1, numsignalsf):
        itersignal[0] = (itersignal[0]+1)%sigperdimf[0]
        for idx9105 in range(1, sigdimensionsf):
            zero = 0
            for idx9106 in range(0, idx9105):
                zero += itersignal[idx9106]
            if zero == 0:
                itersignal[idx9105] = (itersignal[idx9105]+1)%sigperdimf[idx9105]
        signals_indexed[idx9104] = itersignal
        
    assort_array = np.zeros((numsignalsf, numsignalsf), dtype=np.int64)
    for idx9200 in range(0, numsignalsf):
        for idx9201 in range(0, numsignalsf):
            sigsamecount = 0
            for idx9202 in range(0, sigdimensionsf):
                if (signals_indexed[idx9200][idx9202] == signals_indexed[idx9201][idx9202]) and signals_indexed[idx9200][idx9202] != 0:
                    sigsamecount += 1
            assort_array[idx9200][idx9201] = (base_connection_weightsf[sigsamecount])**homophily_factorf
    return assort_array


# takes the assort array and the number of agents using each signal and returns the assort multipliers
@numba.jit(nopython=True, fastmath=False, parallel=False)
def find_assort_multipliers(assort_array, assort_sig_counts, numsignalsf, population_sizef):
    assort_multipliers = np.zeros((numsignalsf, numsignalsf), dtype=np.float64)
    for idx9300 in range(0, numsignalsf):
        amcount = 0
        for idx9301 in range(0, numsignalsf):
            amcount += (assort_array[idx9300][idx9301])*(assort_sig_counts[idx9301])
        assortmod = population_sizef/amcount
        for idx9302 in range(0, numsignalsf):
            assort_multipliers[idx9300][idx9302] = (assort_array[idx9300][idx9302])*assortmod
    return assort_multipliers


# function to take profile counts and return signal counts
@numba.jit(nopython=True, fastmath=False, parallel=False)
def get_sig_counts(numsignalsf, profiles_indexed, numprofilesf, sigmultipliersf, sigdimensionsf, profile_count_untyped):
    assort_sig_counts = np.zeros((numsignalsf), dtype=np.int64) 
    for idx9303 in range(0, numprofilesf):
        if profile_count_untyped[idx9303] != 0:
            asprofile = profiles_indexed[idx9303]
            assig = 0
            for idx9304 in range(0, sigdimensionsf):
                assig += (asprofile[idx9304]*sigmultipliersf[idx9304])
            assig = floor(assig)
            assort_sig_counts[assig] += profile_count_untyped[idx9303]
    return assort_sig_counts


# function to take profile counts and return signal counts
@numba.jit(nopython=True, fastmath=False, parallel=False)
def get_sig_countsRE(numsignalsf, agentprofilepicks, population_sizef, sigmultipliersf, sigdimensionsf):
    assort_sig_counts = np.zeros((numsignalsf), dtype=np.uint16) # uint16 works as long as population is under 65k
    for idx9303 in range(0, population_sizef):
        asprofile = (agentprofilepicks[idx9303]).copy()
        assig = 0
        for idx9304 in range(0, sigdimensionsf):
            assig += (asprofile[idx9304]*sigmultipliersf[idx9304])
        assig = floor(assig)
        assort_sig_counts[assig] += 1
    return assort_sig_counts








@numba.jit(nopython=True, fastmath=False, parallel=False)
def greetings_full_play(numtypesf, numsignalsf, runlengthf, reinforcementf, punishmentf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    # A: first run the full simulation
    for idxN0 in range(0, runlengthf):
        sigurnsf, recurnsf = greetings_single_tstep(reinforcementf, punishmentf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf)
    # B: now lets go from the urn contents to a format in wich results are interpretable
    simple_stat_000, socialsig_aggregate, action_aggregate = greetings_results(poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef)
    return simple_stat_000, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, runid

@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBS_full_play(signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    typed_time = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    signal_time = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    typed_time_norm = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    signal_time_norm = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    snap_length = runlengthf//signal_snapshotsf
#     print(snap_length)
    socialsig_aggregate = np.zeros((signal_snapshotsf, numtypesf, numsignalsf), dtype=np.int64)
    timedex = 0
    snapcount = 0
    itercount = 0
    genBSpunishf_iter = genBSpunishf[itercount]
    # A: first run the full simulation
    for idxN0 in range(0, runlengthf):
        sigurnsf, recurnsf = genBS_single_tstep(genBSpunishf_iter, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf)
        if (idxN0+1) % record_intervalf == 0: # record time series info
            typed_time, signal_time, typed_time_norm, signal_time_norm = time_update(typed_time, signal_time, typed_time_norm, signal_time_norm, poptf, sigurnsf, recurnsf, population_sizef, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigmultipliersf, timedex)
            timedex += 1
        if (idxN0+1) % snap_length == 0:
            itercount = floor((itercount+1)%2)
            genBSpunishf_iter = genBSpunishf[itercount]
            dummy0, socialsig_aggregate[snapcount], dummy1 = genBS_results(numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef)
            snapcount += 1
#             print(f'this one is okay {snapcount} {snap_length} {idxN0+1}')
    # B: now lets go from the urn contents to a format in wich results are interpretable
    simple_stat_001, dummy2, action_aggregate = genBS_results(numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef)
    return simple_stat_001, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid

@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBSreplicator_full_play(mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    typed_time = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    signal_time = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    typed_time_norm = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    signal_time_norm = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    snap_length = runlengthf//signal_snapshotsf
#     print(snap_length)
    socialsig_aggregate = np.zeros((signal_snapshotsf, numtypesf, numsignalsf), dtype=np.int64)
    timedex = 0
    snapcount = 0
    itercount = 0
    genBSpunishf_iter = genBSpunishf[itercount]
    # initialize sig and rec urns for the replicator dynamics
    sigurnsf, recurnsf, profiles_indexed, sigperdimf, profilecaps = replicator_initialize(rngf, repmultipliersf, numprofilesf, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigurnsf, recurnsf, population_sizef)
    # in the first step, convert sigurns and recurns to an easy format for doing the replicator dynamics
#     typestotal, profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicator_first_tstep(repmultipliersf, genBSpunishf_iter, numsignalsf, numprofilesf, numtypesf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, profiles_indexed)
    typestotal, profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicatorexec_k_step(numtypesf, rngf, population_sizef, numprofilesf, poptf)
    agentprofilesCOPY = agentprofiles.copy()
    # ******************************
    mutations10, newprofiles10, muteprofilestrings10 = muterandoms10_smart(rngf, mutateratef, population_sizef, numprofilesf, len(repmultipliersf))
    mute_idxN = 0
    profile_count_typed, profile_count_untyped = repmutation_smart(repmultipliersf, profilecaps, profiles_indexed, numtypesf, numprofilesf, profile_count_typed, profile_count_untyped, mutations10[mute_idxN], newprofiles10[mute_idxN], muteprofilestrings10[mute_idxN])
    mute_idxN += 1
    assort_array = assort_array_create(homophily_factorf, sigperdimf, numsignalsf, sigdimensionsf, base_connection_weightsf)
    # A: first run the full simulation
    idxN0 = 1
    while idxN0 < runlengthf:
        assort_sig_counts = get_sig_counts(numsignalsf, profiles_indexed, numprofilesf, sigmultipliersf, sigdimensionsf, profile_count_untyped)
        assort_multipliers = find_assort_multipliers(assort_array, assort_sig_counts, numsignalsf, population_sizef)
        profile_utility_bytype = utilcompute(assort_multipliers, numtypesf, sigmultipliersf, sigdimensionsf, numprofilesf, profiles_indexed, profile_count_untyped, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf_iter)
        profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicator_single_tstep(numprofilesf, numtypesf, typestotal, profile_count_typed, agentprofiles, profile_utility_bytype, rngf, population_sizef, poptf)
        if (idxN0+1) % record_intervalf == 0: # record time series info
            # function to convert agent profiles into appropriate sig and rec urns
            sigurnsf, recurnsf = repconvert(sigurnsf, recurnsf, agentprofiles, profiles_indexed, sigdimensionsf, numsignalsf, population_sizef)
            typed_time, signal_time, typed_time_norm, signal_time_norm = time_update(typed_time, signal_time, typed_time_norm, signal_time_norm, poptf, sigurnsf, recurnsf, population_sizef, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigmultipliersf, timedex)
            timedex += 1
        if (idxN0) % snap_length == 0:
            itercount = floor((itercount+1)%2)
            genBSpunishf_iter = genBSpunishf[itercount]
            sigurnsf, recurnsf = repconvert(sigurnsf, recurnsf, agentprofiles, profiles_indexed, sigdimensionsf, numsignalsf, population_sizef)
            dummy00, dummy0, socialsig_aggregate[snapcount], dummy1 = genBS_results(numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef)
            snapcount += 1
        # ******************************
        if idxN0%100 == 0:
            if np.array_equal(agentprofilesCOPY, agentprofiles):
                idxN0 = runlengthf
            else:
                agentprofilesCOPY = agentprofiles.copy()
        if idxN0%10 == 0:
            mutations10, newprofiles10, muteprofilestrings10 = muterandoms10_smart(rngf, mutateratef, population_sizef, numprofilesf, len(repmultipliersf))
            mute_idxN = 0
        profile_count_typed, profile_count_untyped = repmutation_smart(repmultipliersf, profilecaps, profiles_indexed, numtypesf, numprofilesf, profile_count_typed, profile_count_untyped, mutations10[mute_idxN], newprofiles10[mute_idxN], muteprofilestrings10[mute_idxN])
        mute_idxN += 1
        idxN0 += 1
    # function to convert agent profiles into appropriate sig and rec urns
    sigurnsf, recurnsf = repconvert(sigurnsf, recurnsf, agentprofiles, profiles_indexed, sigdimensionsf, numsignalsf, population_sizef)
    # B: now lets go from the urn contents to a format in wich results are interpretable
    simple_stat_002, simple_stat_001, socialsig_aggregate[snapcount], action_aggregate = genBS_results(numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef)
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid

@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBSreplicatorexec_full_play(mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    typed_time = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    signal_time = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    typed_time_norm = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    signal_time_norm = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    snap_length = runlengthf//signal_snapshotsf
#     print(snap_length)
    socialsig_aggregate = np.zeros((signal_snapshotsf, numtypesf, numsignalsf), dtype=np.int64)
    timedex = 0
    snapcount = 0
    itercount = 0
    genBSpunishf_iter = genBSpunishf[itercount]
    # initialize sig and rec urns for the replicator dynamics
    sigurnsf, recurnsf, execurnsf, profiles_indexed, sigperdimf, profilecaps = replicatorexec_initialize(epsilonf, rngf, repmultipliersf, numprofilesf, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigurnsf, recurnsf, population_sizef)
#     # in the first step, convert sigurns and recurns to an easy format for doing the replicator dynamics
#     typestotal, profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicatorexec_first_tstep(sigcostf, sigperdimf, repmultipliersf, genBSpunishf_iter, numsignalsf, numprofilesf, numtypesf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, execurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, profiles_indexed)
    typestotal, profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicatorexec_k_step(numtypesf, rngf, population_sizef, numprofilesf, poptf)
    agentprofilesCOPY = agentprofiles.copy()
    mutations10, newprofiles10, muteprofilestrings10 = muterandoms10_smart(rngf, mutateratef, population_sizef, numprofilesf, len(repmultipliersf))
    mute_idxN = 0
    profile_count_typed, profile_count_untyped = repmutation_smart(repmultipliersf, profilecaps, profiles_indexed, numtypesf, numprofilesf, profile_count_typed, profile_count_untyped, mutations10[mute_idxN], newprofiles10[mute_idxN], muteprofilestrings10[mute_idxN])
    mute_idxN += 1
    # A: first run the full simulation
    idxN0 = 1
    assort_array = execassort_array_create(homophily_factorf, sigperdimf, numsignalsf, sigdimensionsf, base_connection_weightsf)
    
    while idxN0 < runlengthf:
        assort_sig_counts = get_sig_counts(numsignalsf, profiles_indexed, numprofilesf, sigmultipliersf, sigdimensionsf, profile_count_untyped)
        assort_multipliers = find_assort_multipliers(assort_array, assort_sig_counts, numsignalsf, population_sizef)
        profile_utility_bytype = executilcompute(assort_multipliers, sigcostf, sigperdimf, numtypesf, sigmultipliersf, sigdimensionsf, numsignalsf, numprofilesf, profiles_indexed, profile_count_untyped, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf_iter)
        profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicator_single_tstep(numprofilesf, numtypesf, typestotal, profile_count_typed, agentprofiles, profile_utility_bytype, rngf, population_sizef, poptf)
        if (idxN0+1) % record_intervalf == 0: # record time series info
            # function to convert agent profiles into appropriate sig and rec urns
            sigurnsf, recurnsf, execurnsf = execrepconvert(sigurnsf, recurnsf, execurnsf, agentprofiles, profiles_indexed, sigdimensionsf, numsignalsf, population_sizef)
            typed_time, signal_time, typed_time_norm, signal_time_norm = time_update(typed_time, signal_time, typed_time_norm, signal_time_norm, poptf, sigurnsf, recurnsf, population_sizef, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigmultipliersf, timedex)
            timedex += 1
        if (idxN0) % snap_length == 0:
            itercount = floor((itercount+1)%2)
            genBSpunishf_iter = genBSpunishf[itercount]
            sigurnsf, recurnsf, execurnsf = execrepconvert(sigurnsf, recurnsf, execurnsf, agentprofiles, profiles_indexed.copy(), sigdimensionsf, numsignalsf, population_sizef)
#             dummy0, socialsig_aggregate[snapcount], dummy1 = genBSexec_results(sigperdimf, profiles_indexed, agentprofiles, numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, execurnsf, population_sizef)
#             sigurnsf, recurnsf = repconvert(sigurnsf, recurnsf, agentprofiles, profiles_indexed, sigdimensionsf, numsignalsf, population_sizef)
            dummy2, dummy0, socialsig_aggregate[snapcount], dummy1 = genBSexec_results(snapcount, sigperdimf.copy(), profiles_indexed.copy(), agentprofiles.copy(), numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf.copy(), recurnsf.copy(), execurnsf.copy(), population_sizef)
            snapcount += 1
        if idxN0%100 == 0:
            if np.array_equal(agentprofilesCOPY, agentprofiles):
                idxN0 = runlengthf
            else:
                agentprofilesCOPY = agentprofiles.copy()
        # lets do mutation after we check for a cycle for obvious reasons (mutating befor the loop check would make the odds of getting out of the loop incredibly low)
        # note that we only need to do mutation on profile_count_typed and profile_count_untyped since agent profiles is repopulated from scratch every timestep
        # skipping agent profiles will also mean that mutation wont effect the final timestep before measuring the results
        if idxN0%10 == 0:
            mutations10, newprofiles10, muteprofilestrings10 = muterandoms10_smart(rngf, mutateratef, population_sizef, numprofilesf, len(repmultipliersf))
            mute_idxN = 0
        profile_count_typed, profile_count_untyped = repmutation_smart(repmultipliersf, profilecaps, profiles_indexed, numtypesf, numprofilesf, profile_count_typed, profile_count_untyped, mutations10[mute_idxN], newprofiles10[mute_idxN], muteprofilestrings10[mute_idxN])
        mute_idxN += 1
        idxN0 += 1
#             print(f'this is not okay {snapcount} {snap_length} {idxN0}')
    # function to convert agent profiles into appropriate sig and rec urns
    sigurnsf, recurnsf, execurnsf = execrepconvert(sigurnsf, recurnsf, execurnsf, agentprofiles, profiles_indexed, sigdimensionsf, numsignalsf, population_sizef)
    # B: now lets go from the urn contents to a format in wich results are interpretable
    simple_stat_002, simple_stat_001, socialsig_aggregate[snapcount], action_aggregate = genBSexec_results(snapcount, sigperdimf, profiles_indexed, agentprofiles, numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, execurnsf, population_sizef)
#     simple_stat_002, simple_stat_001, socialsig_aggregate[snapcount], action_aggregate = genBS_results(numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf, recurnsf, population_sizef)
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, execurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid


@numba.jit(nopython=True, fastmath=False, parallel=False)
def genBS_f7_RothErevExec_full_play(forgetf, mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    typed_time = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    signal_time = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    typed_time_norm = np.zeros((numtypesf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    signal_time_norm = np.zeros((numsignalsf, numsignalsf, numactionsf, runlengthf//record_intervalf), dtype=np.float64)
    snap_length = runlengthf//signal_snapshotsf
    # adding in exec urns which will be sigdimensionsf long with a 0 if the agent does not attend to the dimension and 1 if the agent does, this is basically just usefull for end calculations but not used during actual simulations
    execurnsf = np.zeros((population_sizef, sigdimensionsf), dtype=np.int64)
#     print(snap_length)
    socialsig_aggregate = np.zeros((signal_snapshotsf, numtypesf, numsignalsf), dtype=np.int64)
    timedex = 0
    snapcount = 0
    itercount = 0
    genBSpunishf_iter = genBSpunishf[itercount]
    # initialize sig and rec urns for the replicator dynamics
    sigperdimf, profilecaps = RothErevExec_initialize(epsilonf, rngf, repmultipliersf, numprofilesf, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigurnsf, recurnsf, population_sizef)
    profile_length = len(profilecaps)
    # let's use "agentproilepics" to store the signals and actions in response to signals tha are use for a given timestep
    # we can also check against this for the halt condition instead of agent profiles
    agentprofilepicks = np.zeros((population_sizef, profile_length), dtype=np.uint16)
    agentprofilepicksCOPY = agentprofilepicks.copy()
    picks10 = rngf.random((10, population_sizef, profile_length))
    mute_idxN = 0
    
    # A: first run the full simulation
    idxN0 = 1
    assort_array = execassort_array_create(homophily_factorf, sigperdimf, numsignalsf, sigdimensionsf, base_connection_weightsf)
    
    while idxN0 < runlengthf:
        # need function to go from pics10, mute_idxN, and the urns to agents picks for the next timestep
        agentprofilepicks = random2picks(picks10[mute_idxN], sigdimensionsf, sigurnsf, recurnsf, population_sizef, agentprofilepicks, 
                                        profile_length, profilecaps, numsignalsf)
        mute_idxN += 1
        assort_sig_counts = get_sig_countsRE(numsignalsf, agentprofilepicks, population_sizef, sigmultipliersf, sigdimensionsf)
        assort_multipliers = find_assort_multipliers(assort_array, assort_sig_counts, numsignalsf, population_sizef)
        #************************************************************************************************************
        #*********** (obviously other parts of the code have been changed, but this is the key change) **************
        #******************************** replacing replicator with Roth-Erev: **************************************
        # want function that updates urns based on the payoffs and costs of interacting with everyone (weighted by assortment)
        sigurnsf, recurnsf = executilcomputeREforget(forgetf, agentprofilepicks, assort_multipliers, sigcostf, sigperdimf, sigmultipliersf, sigdimensionsf, numsignalsf, numactionsf, population_sizef, poptf, sigurnsf, recurnsf, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf_iter)
        # _____________________________________ what was replaced: __________________________________________________
        # profile_utility_bytype = executilcompute(assort_multipliers, sigcostf, sigperdimf, numtypesf, sigmultipliersf, sigdimensionsf, numsignalsf, numprofilesf, profiles_indexed, profile_count_untyped, coordination_preferencesf, base_connection_weightsf, homophily_factorf, genBSpunishf_iter)
        # profile_count_typed, profile_count_untyped, agentprofiles = genBSreplicator_single_tstep(numprofilesf, numtypesf, typestotal, profile_count_typed, agentprofiles, profile_utility_bytype, rngf, population_sizef, poptf)
        # ************************* end replacement of replicator ***************************************************
        #************************************************************************************************************
        if (idxN0+1) % record_intervalf == 0: # record time series info
            # function to convert agent profiles into appropriate sig and rec urns
            sigurnsf_4report, recurnsf_4report, execurnsf_4report = execrepconvertRE(sigurnsf, recurnsf, execurnsf, sigdimensionsf, numsignalsf, population_sizef)
            typed_time, signal_time, typed_time_norm, signal_time_norm = time_update(typed_time, signal_time, typed_time_norm, signal_time_norm, poptf, sigurnsf_4report, recurnsf_4report, population_sizef, numsignalsf, numactionsf, numtypesf, sigdimensionsf, sigmultipliersf, timedex)
            timedex += 1
        if (idxN0) % snap_length == 0:
            itercount = floor((itercount+1)%2)
            genBSpunishf_iter = genBSpunishf[itercount]
            sigurnsf_4report, recurnsf_4report, execurnsf_4report = execrepconvertRE(sigurnsf, recurnsf, execurnsf, sigdimensionsf, numsignalsf, population_sizef)
            dummy2, dummy0, socialsig_aggregate[snapcount], dummy1 = genBS_resultsRE(sigperdimf, numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf_4report, recurnsf_4report, population_sizef)
            snapcount += 1
        if idxN0%1000 == 0:
            if np.array_equal(agentprofilepicksCOPY, agentprofilepicks):
                idxN0 = runlengthf
            else:
                agentprofilepicksCOPY = agentprofilepicks.copy()
        if idxN0%10 == 0:
            picks10 = rngf.random((10, population_sizef, profile_length))
            mute_idxN = 0
        idxN0 += 1
#             print(f'this is not okay {snapcount} {snap_length} {idxN0}')
    # function to convert agent profiles into appropriate sig and rec urns
    sigurnsf_4report, recurnsf_4report, execurnsf_4report = execrepconvertRE(sigurnsf, recurnsf, execurnsf, sigdimensionsf, numsignalsf, population_sizef)
    # B: now lets go from the urn contents to a format in wich results are interpretable
    simple_stat_002, simple_stat_001, socialsig_aggregate[snapcount], action_aggregate = genBS_resultsRE(sigperdimf, numactionsf, poptf, sigdimensionsf, sigmultipliersf, numtypesf, numsignalsf, sigurnsf_4report, recurnsf_4report, population_sizef)
    
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurnsf_4report, recurnsf_4report, execurnsf_4report, typed_time, signal_time, typed_time_norm, signal_time_norm, runid

def greetings_full_play_typed(numtypesf, numsignalsf, runlengthf, reinforcementf, punishmentf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    sigurnsf = numba.typed.List(sigurnsf)
    simple_stat_000, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, runid = greetings_full_play(numtypesf, numsignalsf, runlengthf, reinforcementf, punishmentf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid)
    sigurns_untyped = []
    for surn in sigurnsf:
        sigurns_untyped.append(np.array(surn))
    return simple_stat_000, socialsig_aggregate, action_aggregate, sigurns_untyped, recurnsf, runid


def genBS_full_play_typed(signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    sigurnsf = numba.typed.List(sigurnsf)
    simple_stat_001, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid = genBS_full_play(signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid)
    sigurns_untyped = []
    for surn in sigurnsf:
        sigurns_untyped.append(np.array(surn))
    return simple_stat_001, socialsig_aggregate, action_aggregate, sigurns_untyped, recurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid

def genBSreplicator_full_play_typed(mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    sigurnsf = numba.typed.List(sigurnsf)
    simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid = genBSreplicator_full_play(mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid)
    sigurns_untyped = []
    for surn in sigurnsf:
        sigurns_untyped.append(np.array(surn))
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurns_untyped, recurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid

def genBSreplicatorexec_full_play_typed(mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    sigurnsf = numba.typed.List(sigurnsf)
    simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, execurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid = genBSreplicatorexec_full_play(mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid)
    sigurns_untyped = []
    for surn in sigurnsf:
        sigurns_untyped.append(np.array(surn))
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurns_untyped, recurnsf, execurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid

def genBS_f7_RothErevExec_full_play_typed(forgetf, mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid):
    sigurnsf = numba.typed.List(sigurnsf)
    simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurnsf, recurnsf, execurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid = genBS_f7_RothErevExec_full_play(forgetf, mutateratef, sigcostf, repmultipliersf, numprofilesf, signal_snapshotsf, record_intervalf, genBSpunishf, numtypesf, numsignalsf, runlengthf, numactionsf, coordination_preferencesf, poptf, sigurnsf, recurnsf, population_sizef, sigdimensionsf, base_connection_weightsf, sigmultipliersf, homophily_factorf, rngf, epsilonf, runid)
    sigurns_untyped = []
    for surn in sigurnsf:
        sigurns_untyped.append(np.array(surn))
    return simple_stat_002, simple_stat_001, socialsig_aggregate, action_aggregate, sigurns_untyped, recurnsf, execurnsf, typed_time, signal_time, typed_time_norm, signal_time_norm, runid