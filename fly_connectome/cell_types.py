"""
FlyWire Cell Types and Neuropil Taxonomy for Drosophila Connectome Swarm Mesh.
References:
- FlyWire Whole-Brain Connectome (Dorkenwald et al., Schlegel et al., 2024)
- Johnston's Organ subdivisions (Kamikouchi et al., 2006, 2009; Matsuo et al., 2014)
- Thermosensory & Hygrosensory Glomeruli (Frank et al., 2015; Marin et al., 2020)
- Visual Looming & Conspecific Tracking (Klapoetke et al., 2017; Keleş et al., 2020)
- Descending Neurons & Flight Control (Namiki et al., 2018)
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List

class Neuropil(str, Enum):
    AMMC = "AMMC"       # Antennal Mechanosensory and Motor Center
    AL = "AL"           # Antennal Lobe
    WED = "WED"         # Wedge
    LH = "LH"           # Lateral Horn
    CX_EB = "CX_EB"     # Central Complex: Ellipsoid Body
    CX_PB = "CX_PB"     # Central Complex: Protocerebral Bridge
    CX_FB = "CX_FB"     # Central Complex: Fan-shaped Body
    CX_NO = "CX_NO"     # Central Complex: Noduli
    LAL = "LAL"         # Lateral Accessory Lobe
    OPTIC_LP = "OPTIC_LP" # Optic Lobe: Lobula Plate
    OPTIC_LO = "OPTIC_LO" # Optic Lobe: Lobula
    DN_CLUSTER = "DN"   # Descending Neuron Cluster
    VLP = "VLP"         # Ventrolateral Protocerebrum

class Neurotransmitter(str, Enum):
    ACETYLCHOLINE = "ACh"   # Excitatory (+1)
    GABA = "GABA"           # Inhibitory (-1)
    GLUTAMATE = "Glu"       # Frequently inhibitory via GluCl in central circuits (-1)
    HISTAMINE = "HA"        # Photoreceptor inhibitory (-1)
    DOPAMINE = "DA"         # Modulatory
    OCTOPAMINE = "OA"       # Modulatory

class FunctionalModality(str, Enum):
    MECHANOSENSORY_STATIC = "mechano_static"      # Wind vector, barometric pressure (JON-C/E)
    MECHANOSENSORY_ACOUSTIC = "mechano_acoustic"  # Particle velocity, courtship/alarm flutter (JON-A/B)
    THERMOSENSORY_COLD = "thermo_cold"            # Cold drop (VP1l, arista cold cells)
    THERMOSENSORY_WARM = "thermo_warm"            # Warm spike (VP1m, arista warm cells)
    HYGROSENSORY_DRY = "hygro_dry"                # Desiccation / dry air (VP2)
    HYGROSENSORY_MOIST = "hygro_moist"            # Humidity spike (VP3)
    VISUAL_LOOMING = "visual_loom"                # Rapid expansion / approaching neighbor (LC4, LPLC2)
    VISUAL_CONSPECIFIC = "visual_conspecific"      # Neighbor movement tracking (LC11)
    OLFACTORY_ALARM = "olfactory_alarm"           # Drosophila stress odorant (CO2 / dSO via Gr21a/63a)
    CENTRAL_INTEGRATION = "central_integration"   # Compass, heading, valence integration
    MOTOR_DESCENDING = "motor_descending"         # Descending command to ventral nerve cord

@dataclass(frozen=True)
class CellTypeSpec:
    name: str
    neuropil: Neuropil
    modality: FunctionalModality
    neurotransmitter: Neurotransmitter
    default_sign: float  # +1.0 for excitatory, -1.0 for inhibitory
    description: str

# Standard Core Sub-Circuit Neuron Types
CORE_CELL_CATALOG = {
    # --- Mechanosensory / Weather (Wind & Pressure) ---
    "JON_C": CellTypeSpec(
        name="JON_C",
        neuropil=Neuropil.AMMC,
        modality=FunctionalModality.MECHANOSENSORY_STATIC,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Johnston's Organ Zone C: static antennal deflection (steady wind & low-frequency pressure drop)"
    ),
    "JON_E": CellTypeSpec(
        name="JON_E",
        neuropil=Neuropil.AMMC,
        modality=FunctionalModality.MECHANOSENSORY_STATIC,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Johnston's Organ Zone E: sustained deflection & gravity/barometric vector"
    ),

    # --- Mechanosensory / Communication (Near-field Acoustic Particle Velocity) ---
    "JON_A": CellTypeSpec(
        name="JON_A",
        neuropil=Neuropil.AMMC,
        modality=FunctionalModality.MECHANOSENSORY_ACOUSTIC,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Johnston's Organ Zone A: high-frequency antennal vibrations (~200-350 Hz wing buzz)"
    ),
    "JON_B": CellTypeSpec(
        name="JON_B",
        neuropil=Neuropil.AMMC,
        modality=FunctionalModality.MECHANOSENSORY_ACOUSTIC,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Johnston's Organ Zone B: acoustic near-field particle velocity detection"
    ),

    # --- Thermosensory Inputs ---
    "VP1l_Cold": CellTypeSpec(
        name="VP1l_Cold",
        neuropil=Neuropil.AL,
        modality=FunctionalModality.THERMOSENSORY_COLD,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Antennal lobe lateral VP1 projection neuron receiving cold shock from arista (IR21a/25a)"
    ),
    "VP1m_Warm": CellTypeSpec(
        name="VP1m_Warm",
        neuropil=Neuropil.AL,
        modality=FunctionalModality.THERMOSENSORY_WARM,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Antennal lobe medial VP1 projection neuron receiving heat spike from arista (TrpA1)"
    ),

    # --- Hygrosensory Inputs ---
    "VP2_Dry": CellTypeSpec(
        name="VP2_Dry",
        neuropil=Neuropil.AL,
        modality=FunctionalModality.HYGROSENSORY_DRY,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Antennal lobe VP2 projection neuron: humidity drop / dry air shock (IR40a)"
    ),
    "VP3_Moist": CellTypeSpec(
        name="VP3_Moist",
        neuropil=Neuropil.AL,
        modality=FunctionalModality.HYGROSENSORY_MOIST,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Antennal lobe VP3 projection neuron: humidity spike / saturated air (IR68a)"
    ),

    # --- Visual Inputs (Inter-fly communication & conspecific tracking) ---
    "LC11": CellTypeSpec(
        name="LC11",
        neuropil=Neuropil.OPTIC_LO,
        modality=FunctionalModality.VISUAL_CONSPECIFIC,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Lobula columnar neuron 11: detects small moving conspecific targets & rapid neighbor saccades"
    ),
    "LC4": CellTypeSpec(
        name="LC4",
        neuropil=Neuropil.OPTIC_LO,
        modality=FunctionalModality.VISUAL_LOOMING,
        neurotransmitter=Neuropil.OPTIC_LO,
        default_sign=1.0,
        description="Lobula columnar neuron 4: rapid optical looming expansion detector"
    ),

    # --- Olfactory Alarm ---
    "Glomerulus_V_CO2": CellTypeSpec(
        name="Glomerulus_V_CO2",
        neuropil=Neuropil.AL,
        modality=FunctionalModality.OLFACTORY_ALARM,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Glomerulus V projection neuron: receives CO2 / stress odorant from Gr21a/Gr63a sensilla"
    ),

    # --- Intermediate Processing Neuropils ---
    "AMMC_Int1": CellTypeSpec(
        name="AMMC_Int1",
        neuropil=Neuropil.AMMC,
        modality=FunctionalModality.CENTRAL_INTEGRATION,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="AMMC local interneuron relaying mechanical deflection"
    ),
    "WED_PN1": CellTypeSpec(
        name="WED_PN1",
        neuropil=Neuropil.WED,
        modality=FunctionalModality.CENTRAL_INTEGRATION,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Wedge projection neuron 1: translates antennal displacement into directional wind vector"
    ),
    "WED_Acoustic_PN": CellTypeSpec(
        name="WED_Acoustic_PN",
        neuropil=Neuropil.WED,
        modality=FunctionalModality.CENTRAL_INTEGRATION,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Wedge projection neuron relaying near-field wing buzz signals to central brain"
    ),
    "LHON_Aversive": CellTypeSpec(
        name="LHON_Aversive",
        neuropil=Neuropil.LH,
        modality=FunctionalModality.CENTRAL_INTEGRATION,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Lateral Horn Output Neuron: integrates negative valence (cold + dry + CO2 + alert sound)"
    ),
    "CX_EPG_Compass": CellTypeSpec(
        name="CX_EPG_Compass",
        neuropil=Neuropil.CX_EB,
        modality=FunctionalModality.CENTRAL_INTEGRATION,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Central Complex EPG (compass) ring neuron maintaining internal heading representation"
    ),
    "CX_FB_Steering": CellTypeSpec(
        name="CX_FB_Steering",
        neuropil=Neuropil.CX_FB,
        modality=FunctionalModality.CENTRAL_INTEGRATION,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Central Complex Fan-shaped body columnar neuron computing steering vector relative to wind"
    ),
    "LAL_Interneuron": CellTypeSpec(
        name="LAL_Interneuron",
        neuropil=Neuropil.LAL,
        modality=FunctionalModality.CENTRAL_INTEGRATION,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Lateral Accessory Lobe pre-motor relay converging CX and LH outputs into descending drive"
    ),
    "GABA_Refractory_LN": CellTypeSpec(
        name="GABA_Refractory_LN",
        neuropil=Neuropil.LAL,
        modality=FunctionalModality.CENTRAL_INTEGRATION,
        neurotransmitter=Neurotransmitter.GABA,
        default_sign=-1.0,
        description="Broad GABAergic interneuron mediating post-saccadic refractory feedback inhibition"
    ),

    # --- Descending Neurons (Motor & Signaling Outputs) ---
    "DNp01_AlarmSaccade": CellTypeSpec(
        name="DNp01_AlarmSaccade",
        neuropil=Neuropil.DN_CLUSTER,
        modality=FunctionalModality.MOTOR_DESCENDING,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Descending neuron p01: triggers immediate high-g roll/yaw evasive saccade and wing flare"
    ),
    "DNa02_Steering": CellTypeSpec(
        name="DNa02_Steering",
        neuropil=Neuropil.DN_CLUSTER,
        modality=FunctionalModality.MOTOR_DESCENDING,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Descending neuron a02: commands sustained asymmetrical wing stroke amplitude for heading change"
    ),
    "bLN1_AcousticBuzz": CellTypeSpec(
        name="bLN1_AcousticBuzz",
        neuropil=Neuropil.DN_CLUSTER,
        modality=FunctionalModality.MOTOR_DESCENDING,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Descending motor modifier: drives rapid wing oscillation pulse (acoustic warning burst)"
    ),
    "DN_OdorRelease": CellTypeSpec(
        name="DN_OdorRelease",
        neuropil=Neuropil.DN_CLUSTER,
        modality=FunctionalModality.MOTOR_DESCENDING,
        neurotransmitter=Neurotransmitter.ACETYLCHOLINE,
        default_sign=1.0,
        description="Descending motor command driving abdominal spiracle venting of alarm odorant (dSO)"
    ),
}
