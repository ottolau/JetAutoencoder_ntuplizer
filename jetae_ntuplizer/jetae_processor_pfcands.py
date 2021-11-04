from collections import defaultdict
from coffea import hist, processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
import numba as nb
from fast_histogram import histogram1d, histogram2d
import matplotlib.pyplot as plt

def np_acc_int():
    return processor.column_accumulator(np.array([], dtype=np.int64))

def np_acc_float():
    return processor.column_accumulator(np.array([], dtype=np.float64))

def np_acc_image(image_shape):
    return processor.column_accumulator(np.empty((0,)+image_shape, dtype=np.float64))

def normalize(branch):
    #return processor.column_accumulator(ak.to_numpy(ak.fill_none(ak.flatten(branch, axis=None), np.nan)))
    #return processor.column_accumulator(ak.to_numpy(ak.fill_none(branch, -99)))
    return processor.column_accumulator(ak.to_numpy(ak.fill_none(ak.flatten(branch), 0.0)))

def fill_branch(branch, branch_name):
    try:
        tofill_branch = normalize(branch[branch_name])
    except ValueError:
        tofill_branch = None
    return tofill_branch

@nb.jit
def match_one_associatedobj(baseobj_arrayshape, associatedobj_idx, builder):
    for i in range(len(associatedobj_idx)):
        associatedobj = associatedobj_idx[i]
        builder.begin_list()
        tmp_baseobj_idx = []
        tmp_associatedobj_idx = []
        for j in range(len(associatedobj)):
            baseobj_idx = associatedobj[j]
            tmp_baseobj_idx.append(baseobj_idx)
            tmp_associatedobj_idx.append(j)
        for k in range(len(baseobj_arrayshape[i])):
            if k in tmp_baseobj_idx:
                builder.integer(tmp_associatedobj_idx[tmp_baseobj_idx.index(k)])
            else:
                builder.integer(-1)
        builder.end_list()
    return builder


#@nb.jit
#def match_all_associatedobj(baseobj_arrayshape, associatedobj_idx, builder):
#    # event loop
#    for i in range(len(baseobj_arrayshape)):
#        baseobj = baseobj_arrayshape[i]
#        associatedobj = associatedobj_idx[i]
#        builder.begin_list()
#        # fatjet object loop
#        for j in range(len(baseobj)):
#            builder.begin_list()
#            # pf cands index loop
#            for k in range(len(associatedobj)):
#                if associatedobj[k] == j:
#                    builder.integer(k)
#            builder.end_list()
#        builder.end_list()
#    return builder


@nb.jit
def match_fatjet_pfcand(fatjets, fatjetpfcands, pfcands, builder):
    # event loop
    for i in range(len(fatjets)):
        fatjet = fatjets[i]
        fatjetpfcand = fatjetpfcands[i]
        pfcand = pfcands[i]
        builder.begin_list()
        #print("Event: {}".format(i))
        # fatjet object loop
        for j in range(len(fatjet)):
            builder.begin_list()
            #print("\t Jet Idx: {}, Jet eta:{}, Jet phi:{}".format(j, fatjet[j]['eta'], fatjet[j]['phi']))
            # pf cands loop
            for k in range(len(fatjetpfcand)):
                if fatjetpfcand[k]["jetIdx"] == j:
                    builder.integer(fatjetpfcand[k]["pFCandsIdx"])
                    #if i==0:
                    #print("\t\t FatJetPFcand Idx: {},  FatJet Idx: {}, PFcand Idx: {}, PFcand eta: {}, PFcand phi: {}".format(k, fatjetpfcand[k]["jetIdx"], fatjetpfcand[k]["pFCandsIdx"], pfcand[fatjetpfcand[k]["pFCandsIdx"]]["eta"], pfcand[fatjetpfcand[k]["pFCandsIdx"]]["phi"]))
            builder.end_list()
        builder.end_list()
    return builder

def histo_pfcand(fatjetpfcands, hist_range, bins, pfcnad_vars):
    output = defaultdict(list)
    # event loop
    for i in range(len(fatjetpfcands)):
        #if i%100 == 0:
        #    print(i)
        pfcands = fatjetpfcands[i]
        for var in pfcnad_vars:
            output[var].append(histogram2d(ak.to_numpy(pfcands['delta_eta']), ak.to_numpy(pfcands['delta_phi']), range=hist_range, bins=bins, weights=ak.to_numpy(pfcands[var])))
            output['detframe_{}'.format(var)].append(histogram2d(ak.to_numpy(pfcands['eta']), ak.to_numpy(pfcands['phi']), range=hist_range, bins=bins, weights=ak.to_numpy(pfcands[var])))
    return {var: np.array(result) for var, result in output.items()}

class JetAEProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator()
        self._accumulator["tree"] = processor.dict_accumulator()

        # output branches

        # variables: general
        #self._accumulator["tree"]["FatJet_nFatJetPFCands"] = np_acc_int()
        self._accumulator["tree"]["PFCands_delta_eta"] = np_acc_float()
        self._accumulator["tree"]["PFCands_delta_phi"] = np_acc_float()

        # image

        self._accumulator["image"] = processor.dict_accumulator()
        eta_min, eta_max = -2.0, 2.0
        phi_min, phi_max = -3.2, 3.2
        incr = 0.02
        self._hist_range = [[eta_min, eta_max], [phi_min, phi_max]]
        self._eta_bins = np.arange(eta_min, eta_max, incr)
        self._phi_bins = np.arange(phi_min, phi_max, incr)
        self._image_shape = (self._eta_bins.shape[0], self._phi_bins.shape[0])

        self._accumulator["image"]["PFCands_pt"] = np_acc_image(self._image_shape)
        self._accumulator["image"]["PFCands_detframe_pt"] = np_acc_image(self._image_shape)

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        #isRealData = not hasattr(events, "genWeight")

        output = self.accumulator.identity()

        fatjets = events.FatJet
        fatjets_arrayshape = ak.full_like(fatjets.pt, -1)
        #fatjets['svIdx'] = match_one_associatedobj(fatjets_arrayshape, events.FatJetSVs['jetIdx'], ak.ArrayBuilder()).snapshot()
        #fatjets['pfcandIdx'] = match_all_associatedobj(fatjets_arrayshape, events.FatJetPFCands['jetIdx'], ak.ArrayBuilder()).snapshot()
        fatjets['pfcandIdx'] = match_fatjet_pfcand(fatjets, events.FatJetPFCands, events.PFCands, ak.ArrayBuilder()).snapshot()
        #fatjets.add_attributes(
        #    pfcandIdx = match_fatjet_pfcand(fatjets, events.FatJetPFCands, events.PFCands, ak.ArrayBuilder()).snapshot()
        #)
        #print(fatjets.pfcandIdx[0])

        # remove overlapping jet and leptons

        electrons_veto = events.Electron
        electrons_veto = electrons_veto[electrons_veto.pt > 20.0]
        electrons_veto = fatjets.nearest(electrons_veto)
        # accept jet that doesn't have an electron nearby
        electrons_veto_selection = ak.fill_none(fatjets.delta_r(electrons_veto) > 0.4, True)
        fatjets = fatjets[electrons_veto_selection]

        muons_veto = events.Muon
        muons_veto = muons_veto[muons_veto.pt > 20.0]
        muons_veto = fatjets.nearest(muons_veto)
        # accept jet that doesn't have a muon nearby
        muons_veto_selection = ak.fill_none(fatjets.delta_r(muons_veto) > 0.4, True)
        fatjets = fatjets[muons_veto_selection]


        # gen-match
        fatjets = fatjets[~ak.is_none(fatjets.matched_gen, axis=1)]
        fatjets = fatjets[fatjets.delta_r(fatjets.matched_gen) < 0.4]

        # pre-selection
        selections = {}
        selections['pt'] = fatjets.pt > 200.0
        selections['eta'] = abs(fatjets.eta) < 2.0
        #selections['hadronFlavour'] = fatjets.matched_gen.hadronFlavour == 0 # jet not originated from b or c
        selections['all'] = selections['pt'] & selections['eta'] #& selections['hadronFlavour']

        fatjets = fatjets[selections['all']]

        #fatjets = fatjets[ak.num(fatjets) > 0]
        #fatjets = fatjets[ak.argsort(fatjets.pt, axis=1)]
        fatjets = ak.firsts(fatjets)

        #print(fatjets['pfcandIdx'][0])

        gen_fatjets = fatjets.matched_gen
        #subjets_1 = fatjets.subjets[:,0]
        #subjets_2 = fatjets.subjets[:,1]
        #fatjetsvs = events.FatJetSVs._apply_global_index(fatjets.svIdx)

        fatjetpfcands = events.PFCands._apply_global_index(fatjets.pfcandIdx)
        #print(fatjetpfcands[0])
        fatjetpfcands['delta_phi'] = fatjetpfcands.delta_phi(fatjets)
        fatjetpfcands['delta_eta'] = fatjetpfcands['eta'] - fatjets['eta']
        fatjetpfcands['delta_r'] = fatjetpfcands.delta_r(fatjets)

        #print(fatjetpfcands['delta_phi'][0])
        #print(fatjetpfcands['delta_eta'][0])
        #print(fatjetpfcands['delta_r'][0])

        #pfcnad_vars = ['pt']
        #test = histo_pfcand(fatjetpfcands[:100], self._hist_range, self._image_shape, pfcnad_vars)
        #output["image"]["PFCands_pt"] += processor.column_accumulator(test["pt"])
        #output["image"]["PFCands_detframe_pt"] += processor.column_accumulator(test["detframe_pt"])

        #fatjetpfcands = fatjetpfcands[fatjetpfcands.pt > 1.0]  
        #index = 0
        #print(fatjets.eta[index], fatjetpfcands.eta[index], len(fatjetpfcands.eta[index]), ak.num(fatjets['pfcandIdx'], axis=-1)[index], fatjetpfcands.delta_r(fatjets)[index])
        #print(fatjetpfcands_raw.pt, fatjetpfcands.pt)

        #pfcands_max_delta_r = ak.max(fatjetpfcands.delta_r(fatjets), axis=-1)
        #pfcands_mean_delta_r = ak.mean(fatjetpfcands.delta_r(fatjets), axis=-1)
        #pfcands_mean_delta_phi = ak.mean(fatjetpfcands['delta_phi'], axis=-1)
        #pfcands_mean_delta_eta = ak.mean(fatjetpfcands['delta_eta'], axis=-1)

        # fill output branches

        # try to fill default branches
        for branch_name in output["tree"].keys():
          if 'FatJet_gen' in branch_name:
            tofill_branch = fill_branch(gen_fatjets, branch_name.replace('FatJet_gen_', ''))
          elif 'FatJet_subjet1' in branch_name:
            tofill_branch = fill_branch(subjets_1, branch_name.replace('FatJet_subjet1_', ''))
          elif 'FatJet_subjet2' in branch_name:
            tofill_branch = fill_branch(subjets_2, branch_name.replace('FatJet_subjet2_', ''))
          elif 'FatJet_sv' in branch_name:
            tofill_branch = fill_branch(fatjetsvs, branch_name.replace('FatJet_sv_', ''))
          elif 'PFCands_' in branch_name:
            tofill_branch = fill_branch(fatjetpfcands, branch_name.replace('PFCands_', ''))

          else:
            tofill_branch = fill_branch(fatjets, branch_name.replace('FatJet_', ''))
            
          if tofill_branch is not None:
            output["tree"][branch_name] += tofill_branch

        # fill new added branches
        #output["tree"]['FatJet_nFatJetPFCands'] += normalize(ak.num(fatjets['pfcandIdx'], axis=-1))
        #output["tree"]['FatJet_pfcand_max_delta_r'] += normalize(pfcands_max_delta_r)
        #output["tree"]['FatJet_pfcand_mean_delta_r'] += normalize(pfcands_mean_delta_r)
        #output["tree"]['FatJet_pfcand_mean_delta_phi'] += normalize(pfcands_mean_delta_phi)
        #output["tree"]['FatJet_pfcand_mean_delta_eta'] += normalize(pfcands_mean_delta_eta)

        return output


    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":
    import uproot3
    import argparse
    parser = argparse.ArgumentParser(description="Ntuplizer for Jet autoencoder")
    parser.add_argument("--datasets", "-d", type=str, help="List of datasets to run (comma-separated)")
    parser.add_argument("--outputname", "-o", type=str, default='test', help="Name of output files")
    parser.add_argument("--workers", "-w", type=int, default=12, help="Number of workers")
    parser.add_argument("--condor", action="store_true", help="Flag for running on condor")
    args = parser.parse_args()


    fname = "/eos/user/k/klau/pfnano/QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8/nano_mc2018_12_a677915bd61e6c9ff968b87c36658d9d_101.root"
    samples = {
        "QCD_HT500to700": [fname]
    }

    output = processor.run_uproot_job(
        samples,
        treename="Events",
        processor_instance=JetAEProcessor(),
        #executor=processor.futures_executor,
        executor=processor.iterative_executor,
        executor_args={'workers': args.workers, 
                       'schema': PFNanoAODSchema,
                       },
        chunksize=50000,
    )

    #print(output["image"]["PFCands_pt"].value[0])
    #eta_min, eta_max = -2.0, 2.0
    #phi_min, phi_max = -3.2, 3.2
    #incr = 0.02
    #eta_bins = np.arange(eta_min, eta_max, incr)
    #phi_bins = np.arange(phi_min, phi_max, incr)
  
    #for i, image in enumerate(output["image"]["PFCands_pt"].value[5: 15]):
    #    fig, ax = plt.subplots()
    #    plt.imshow(image, origin='lower', interpolation='none', vmin=0, extent=[eta_bins[0], eta_bins[-1], phi_bins[0], phi_bins[-1]])
    #    plt.colorbar()
    #    ax.set_xlabel('eta')
    #    ax.set_ylabel('phi')
    #    fig.savefig('PFCands_pt_{}.pdf'.format(i), bbox_inches='tight')

    ##for i, image in enumerate(output["image"]["PFCands_detframe_pt"].value[5: 15]):
    #    fig, ax = plt.subplots()
    #    plt.imshow(image, origin='lower', interpolation='none', vmin=0, extent=[eta_bins[0], eta_bins[-1], phi_bins[0], phi_bins[-1]])
    #    plt.colorbar()
    #    ax.set_xlabel('eta')
    #    ax.set_ylabel('phi')
    #    fig.savefig('PFCands_detframe_pt_{}.pdf'.format(i), bbox_inches='tight')

    branches = {k: v.value for k, v in output["tree"].items()}
    branches_init = {k: v.value.dtype for k, v in output["tree"].items()}
    #print([(k, len(v)) for k, v in branches.items()])

    with uproot3.recreate(args.outputname.replace('.root','')+'.root') as f:
        f["tree"] = uproot3.newtree(branches_init)
        f["tree"].extend(branches)


