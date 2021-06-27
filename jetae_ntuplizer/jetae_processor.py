from collections import defaultdict
from coffea import hist, processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
import numba as nb

def np_acc_int():
    return processor.column_accumulator(np.array([], dtype=np.int64))

def np_acc_float():
    return processor.column_accumulator(np.array([], dtype=np.float64))

def normalize(branch):
    #return processor.column_accumulator(ak.to_numpy(ak.fill_none(ak.flatten(branch, axis=None), np.nan)))
    return processor.column_accumulator(ak.to_numpy(ak.fill_none(branch, -99)))

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

@nb.jit
def match_all_associatedobj(baseobj_arrayshape, associatedobj_idx, builder):
    for i in range(len(baseobj_arrayshape)):
        baseobj = baseobj_arrayshape[i]
        associatedobj = associatedobj_idx[i]
        builder.begin_list()
        for j in range(len(baseobj)):
            builder.begin_list()
            for k in range(len(associatedobj)):
                if associatedobj[k] == j:
                    builder.integer(k)
            builder.end_list()
        builder.end_list()
    return builder

class JetAEProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator()

        # output branches

        # variables: general
        self._accumulator["FatJet_pt"] = np_acc_float()
        self._accumulator["FatJet_eta"] = np_acc_float()
        self._accumulator["FatJet_phi"] = np_acc_float()
        self._accumulator["FatJet_DDX_jetNSecondaryVertices"] = np_acc_int()
        self._accumulator["FatJet_DDX_jetNTracks"] = np_acc_int()
        self._accumulator["FatJet_DDX_z_ratio"] = np_acc_float()
        self._accumulator["FatJet_Proba"] = np_acc_float()
        self._accumulator["FatJet_area"] = np_acc_float()
        self._accumulator["FatJet_jetId"] = np_acc_int()
        self._accumulator["FatJet_lsf3"] = np_acc_float()
        self._accumulator["FatJet_mass"] = np_acc_float()
        self._accumulator["FatJet_msoftdrop"] = np_acc_float()
        self._accumulator["FatJet_rawFactor"] = np_acc_float()
        self._accumulator["FatJet_n2b1"] = np_acc_float()
        self._accumulator["FatJet_n3b1"] = np_acc_float()
        #self._accumulator["FatJet_nBHadrons"] = np_acc_int()
        #self._accumulator["FatJet_nCHadrons"] = np_acc_int()

        # variables: tau1
        self._accumulator["FatJet_tau1"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_flightDistance2dSig"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_trackEtaRel_0"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_trackEtaRel_1"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_trackEtaRel_2"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_trackSip3dSig_0"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_trackSip3dSig_1"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_vertexDeltaR"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_vertexEnergyRatio"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau1_vertexMass"] = np_acc_float()

        # variables: tau2
        self._accumulator["FatJet_tau2"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau2_flightDistance2dSig"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau2_trackEtaRel_0"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau2_trackEtaRel_1"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau2_trackEtaRel_3"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau2_trackSip3dSig_0"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau2_trackSip3dSig_1"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau2_vertexEnergyRatio"] = np_acc_float()
        self._accumulator["FatJet_DDX_tau2_vertexMass"] = np_acc_float()

        # variables: tau3 and tau4
        self._accumulator["FatJet_tau3"] = np_acc_float()
        self._accumulator["FatJet_tau4"] = np_acc_float()

        # variables: track
        self._accumulator["FatJet_DDX_trackSip2dSigAboveBottom_0"] = np_acc_float()
        self._accumulator["FatJet_DDX_trackSip2dSigAboveBottom_1"] = np_acc_float()
        self._accumulator["FatJet_DDX_trackSip2dSigAboveCharm"] = np_acc_float()
        self._accumulator["FatJet_DDX_trackSip3dSig_0"] = np_acc_float()
        self._accumulator["FatJet_DDX_trackSip3dSig_1"] = np_acc_float()
        self._accumulator["FatJet_DDX_trackSip3dSig_2"] = np_acc_float()
        self._accumulator["FatJet_DDX_trackSip3dSig_3"] = np_acc_float()

        # variables: subjet 1
        self._accumulator["FatJet_subjet1_pt"] = np_acc_float()
        self._accumulator["FatJet_subjet1_eta"] = np_acc_float()
        self._accumulator["FatJet_subjet1_phi"] = np_acc_float()
        self._accumulator["FatJet_subjet1_Proba"] = np_acc_float()
        self._accumulator["FatJet_subjet1_mass"] = np_acc_float()
        self._accumulator["FatJet_subjet1_tau1"] = np_acc_float()
        self._accumulator["FatJet_subjet1_tau2"] = np_acc_float()
        self._accumulator["FatJet_subjet1_tau3"] = np_acc_float()
        self._accumulator["FatJet_subjet1_tau4"] = np_acc_float()
        self._accumulator["FatJet_subjet1_n2b1"] = np_acc_float()
        self._accumulator["FatJet_subjet1_n3b1"] = np_acc_float()

        # variables: subjet 2
        self._accumulator["FatJet_subjet2_pt"] = np_acc_float()
        self._accumulator["FatJet_subjet2_eta"] = np_acc_float()
        self._accumulator["FatJet_subjet2_phi"] = np_acc_float()
        self._accumulator["FatJet_subjet2_Proba"] = np_acc_float()
        self._accumulator["FatJet_subjet2_mass"] = np_acc_float()
        self._accumulator["FatJet_subjet2_tau1"] = np_acc_float()
        self._accumulator["FatJet_subjet2_tau2"] = np_acc_float()
        self._accumulator["FatJet_subjet2_tau3"] = np_acc_float()
        self._accumulator["FatJet_subjet2_tau4"] = np_acc_float()
        self._accumulator["FatJet_subjet2_n2b1"] = np_acc_float()
        self._accumulator["FatJet_subjet2_n3b1"] = np_acc_float()

        self._accumulator["FatJet_hadronFlavour"] = np_acc_int()

        # variables: fatjet sv
        self._accumulator["FatJet_sv_costhetasvpv"] = np_acc_float()
        self._accumulator["FatJet_sv_d3dsig"] = np_acc_float()
        self._accumulator["FatJet_sv_deltaR"] = np_acc_float()
        self._accumulator["FatJet_sv_dxysig"] = np_acc_float()
        self._accumulator["FatJet_sv_enration"] = np_acc_float()
        self._accumulator["FatJet_sv_mass"] = np_acc_float()
        self._accumulator["FatJet_sv_normchi2"] = np_acc_float()
        self._accumulator["FatJet_sv_ntracks"] = np_acc_float()
        self._accumulator["FatJet_sv_phirel"] = np_acc_float()
        self._accumulator["FatJet_sv_pt"] = np_acc_float()
        self._accumulator["FatJet_sv_ptrel"] = np_acc_float()

        self._accumulator["FatJet_nFatJetPFCands"] = np_acc_int()
        self._accumulator["FatJet_pfcand_max_deltar"] = np_acc_float()
        self._accumulator["FatJet_pfcand_mean_deltar"] = np_acc_float()

        # variables: generator level
        self._accumulator["FatJet_gen_pt"] = np_acc_float()
        self._accumulator["FatJet_gen_eta"] = np_acc_float()
        self._accumulator["FatJet_gen_phi"] = np_acc_float()
        self._accumulator["FatJet_gen_hadronFlavour"] = np_acc_int()

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        #isRealData = not hasattr(events, "genWeight")

        output = self.accumulator.identity()

        fatjets = events.FatJet
        fatjets_arrayshape = ak.full_like(fatjets.pt, -1)
        fatjets['svIdx'] = match_one_associatedobj(fatjets_arrayshape, events.FatJetSVs['jetIdx'], ak.ArrayBuilder()).snapshot()
        fatjets['pfcandIdx'] = match_all_associatedobj(fatjets_arrayshape, events.FatJetPFCands['jetIdx'], ak.ArrayBuilder()).snapshot()

        # gen-match
        fatjets = fatjets[~ak.is_none(fatjets.matched_gen)]
        fatjets = fatjets[fatjets.delta_r(fatjets.matched_gen) < 0.4]

        # pre-selection
        selections = {}
        selections['pt'] = fatjets.pt > 200.0
        selections['eta'] = abs(fatjets.eta) < 2.0
        selections['hadronFlavour'] = fatjets.matched_gen.hadronFlavour == 0 # jet not originated from b or c
        selections['all'] = selections['pt'] & selections['eta'] & selections['hadronFlavour']

        fatjets = fatjets[selections['all']]

        fatjets = fatjets[ak.num(fatjets) > 0]
        fatjets = fatjets[ak.argsort(fatjets.pt, axis=1)]
        fatjets = ak.firsts(fatjets)

        gen_fatjets = fatjets.matched_gen
        subjets_1 = fatjets.subjets[:,0]
        subjets_2 = fatjets.subjets[:,1]
        fatjetsvs = events.FatJetSVs._apply_global_index(fatjets.svIdx)
        fatjetpfcands = events.PFCands._apply_global_index(events.FatJetPFCands._apply_global_index(fatjets.pfcandIdx).pFCandsIdx)
        pfcands_max_deltar = ak.max(fatjetpfcands.delta_r(fatjets), axis=-1)
        pfcands_mean_deltar = ak.mean(fatjetpfcands.delta_r(fatjets), axis=-1)

        # fill output branches

        # try to fill default branches
        for branch_name in output.keys():
          if 'FatJet_gen' in branch_name:
            tofill_branch = fill_branch(gen_fatjets, branch_name.replace('FatJet_gen_', ''))
          elif 'FatJet_subjet1' in branch_name:
            tofill_branch = fill_branch(subjets_1, branch_name.replace('FatJet_subjet1_', ''))
          elif 'FatJet_subjet2' in branch_name:
            tofill_branch = fill_branch(subjets_2, branch_name.replace('FatJet_subjet2_', ''))
          elif 'FatJet_sv' in branch_name:
            tofill_branch = fill_branch(fatjetsvs, branch_name.replace('FatJet_sv_', ''))
          else:
            tofill_branch = fill_branch(fatjets, branch_name.replace('FatJet_', ''))
            
          if tofill_branch is not None:
            output[branch_name] += tofill_branch

        # fill new added branches
        output['FatJet_nFatJetPFCands'] = normalize(ak.num(fatjets['pfcandIdx'], axis=-1))
        output['FatJet_pfcand_max_deltar'] = normalize(pfcands_max_deltar)
        output['FatJet_pfcand_mean_deltar'] = normalize(pfcands_mean_deltar)

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
        executor=processor.futures_executor,
        #executor=processor.iterative_executor,
        executor_args={'workers': args.workers, 
                       'schema': PFNanoAODSchema,
                       },
        chunksize=50000,
    )

    branches = {k: v.value for k, v in output.items()}
    branches_init = {k: v.value.dtype for k, v in output.items()}

    with uproot3.recreate(args.outputname.replace('.root','')+'.root') as f:
        f["tree"] = uproot3.newtree(branches_init)
        f["tree"].extend(branches)


