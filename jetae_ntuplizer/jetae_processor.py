from collections import defaultdict
from coffea import hist, processor
from coffea.nanoevents import NanoEventsFactory, PFNanoAODSchema
import numpy as np
import awkward as ak
import uproot3

def np_acc_int():
    return processor.column_accumulator(np.array([], dtype=np.int64))

def np_acc_float():
    return processor.column_accumulator(np.array([], dtype=np.float64))


class JetAEProcessor(processor.ProcessorABC):
    def __init__(self):
        self._accumulator = processor.dict_accumulator()

        # output branches
        #self._accumulator["Jet_pt"] = processor.defaultdict_accumulator(float)
        self._accumulator["Jet_pt"] = np_acc_float()

    @property
    def accumulator(self):
        return self._accumulator

    def process(self, events):
        dataset = events.metadata['dataset']
        #isRealData = not hasattr(events, "genWeight")

        output = self.accumulator.identity()

        jets = events.Jet
        jets = jets[~ak.is_none(jets.matched_gen)]
        jets = jets[jets.delta_r(jets.matched_gen) < 0.4]

        gen_jets = jets.matched_gen
        matched_electrons = jets.matched_electrons                        
        matched_muons = jets.matched_muons                        

        def normalize(branch):
            return processor.column_accumulator(ak.to_numpy(ak.fill_none(ak.flatten(branch), np.nan)))

        output['Jet_pt'] += normalize(jets['pt'])

        return output


    def postprocess(self, accumulator):
        return accumulator


if __name__ == "__main__":

    fname = "/eos/user/k/klau/pfnano/QCD_HT500to700_TuneCP5_13TeV-madgraphMLM-pythia8/nano_mc2018_12_a677915bd61e6c9ff968b87c36658d9d_101.root"
    samples = {
        "DrellYan": [fname]
    }

    output = processor.run_uproot_job(
        samples,
        treename="Events",
        processor_instance=JetAEProcessor(),
        executor=processor.futures_executor,
        executor_args={'workers': 12, 
                       'schema': PFNanoAODSchema,
                       },
        chunksize=50000,
    )

    branches = {k: v.value for k, v in output.items()}
    branches_init = {k: v.value.dtype for k, v in output.items()}

    with uproot3.recreate("test.root") as f:
        f["tree"] = uproot3.newtree(branches_init)
        f["tree"].extend(branches)


