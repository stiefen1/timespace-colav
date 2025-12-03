from enum import Enum
from typing import Tuple, Dict
import numpy as np, logging
from colav.utils.math import wrap_angle_to_pmpi, DEG2RAD, RAD2DEG
from colav.obstacles.moving import MovingShip
logger = logging.getLogger(__name__)

class Encounter(Enum):
    INVALID = 0
    HEAD_ON = 1
    STARBOARD = 2
    PORT = 3
    OVERTAKING = 4

class Recommendation(Enum):
    INVALID = 0
    TURN_RIGHT = 1
    DO_NOTHING = 2
    TURN_LEFT = 3 # in case of good seamanship when crossing a TSS for instance

CROSSING_ANGLE_DEG = 22.5 # Angle w.r.t horizontal that separate starboard, port from overtaking (degrees)

def get_encounter(
        os: MovingShip, 
        ts: MovingShip,
        head_on_lim_deg:float=10,
        ) -> Encounter:
    """
    Returns encounter based on OS and TS pose. 
    
    """
    heading_os_rad = wrap_angle_to_pmpi(DEG2RAD(os.psi) if os.degrees else os.psi, degrees=False)
    rel_xy = np.array([ts.position[0] - os.position[0], ts.position[1] - os.position[1]])
    
    encounter_angle_raw_rad = np.sign(rel_xy[0]) * np.arccos(rel_xy[1] / np.linalg.norm(rel_xy)) - heading_os_rad
    encounter_angle_raw_deg = RAD2DEG(encounter_angle_raw_rad)
    encounter_angle_deg = wrap_angle_to_pmpi(encounter_angle_raw_deg, degrees=True) # [0, 1] @ pose_rel / (norm([0, 1]) * norm(pose_rel)) = cos(a)

    encounter = Encounter.INVALID
    if -head_on_lim_deg <= encounter_angle_deg <= head_on_lim_deg:
        encounter = Encounter.HEAD_ON
    elif -90-CROSSING_ANGLE_DEG <= encounter_angle_deg <= -head_on_lim_deg:
        encounter = Encounter.PORT
    elif head_on_lim_deg <= encounter_angle_deg <= 90+CROSSING_ANGLE_DEG:
        encounter = Encounter.STARBOARD
    elif (-180 <= encounter_angle_deg <= -90-CROSSING_ANGLE_DEG) or (90+CROSSING_ANGLE_DEG <= encounter_angle_deg <= 180):
        encounter = Encounter.OVERTAKING
    else:
        logger.warning(f"Invalid value for encounter with relative angle {encounter_angle_deg:.2f} deg")

    logger.info(f"{encounter.name} encounter detected")
    return encounter

def get_recommendation_from_encounters(ts_wrt_os:Encounter, os_wrt_ts:Encounter, good_seamanship:bool=False, ts_in_TSS:bool=False, os_in_TSS:bool=False) -> Recommendation:
    assert ts_wrt_os != Encounter.INVALID, f"Encounter ts_wrt_os is invalid"
    assert os_wrt_ts != Encounter.INVALID, f"Encounter os_wrt_ts is invalid"
    
    if ts_wrt_os == Encounter.OVERTAKING or os_wrt_ts == Encounter.OVERTAKING:
        recommendation = Recommendation.DO_NOTHING
    
    elif good_seamanship and ts_in_TSS and (not os_in_TSS):
        match ts_wrt_os:
            case Encounter.HEAD_ON:
                    if os_wrt_ts == Encounter.STARBOARD:
                        recommendation = Recommendation.TURN_LEFT
                    else:
                        recommendation = Recommendation.TURN_RIGHT
            case Encounter.STARBOARD:
                if os_wrt_ts == Encounter.STARBOARD:
                    recommendation = Recommendation.TURN_LEFT
                else:
                    recommendation = Recommendation.TURN_RIGHT
            case Encounter.PORT:
                if os_wrt_ts == Encounter.PORT:
                    recommendation = Recommendation.TURN_RIGHT
                else:
                    recommendation = Recommendation.TURN_LEFT
            case _:
                logger.warning(f"Invalid Encounter value {ts_wrt_os}")
                recommendation = Recommendation.INVALID
    else:
        match ts_wrt_os:
            case Encounter.HEAD_ON:
                    recommendation = Recommendation.TURN_RIGHT
            case Encounter.STARBOARD:
                if os_wrt_ts == Encounter.STARBOARD:
                    recommendation = Recommendation.DO_NOTHING
                else:
                    recommendation = Recommendation.TURN_RIGHT
            case Encounter.PORT:
                if os_wrt_ts == Encounter.PORT:
                    recommendation = Recommendation.DO_NOTHING
                else:
                    recommendation = Recommendation.TURN_RIGHT
            case _:
                logger.warning(f"Invalid Encounter value {ts_wrt_os}")
                recommendation = Recommendation.INVALID

    logger.info(f"Recommended {recommendation.name}")
    return recommendation

def get_recommendation_for_os(
        os: MovingShip, 
        ts: MovingShip,
        head_on_lim_deg: float = 10,
        good_seamanship: bool = False,
        ts_in_TSS: bool = False,
        os_in_TSS: bool = False
    ) -> Tuple[Recommendation, Dict]:
    """
    
    """
    encounter_ts_wrt_os = get_encounter(os, ts, head_on_lim_deg=head_on_lim_deg)
    encounter_os_wrt_ts = get_encounter(ts, os, head_on_lim_deg=head_on_lim_deg)
    info = {
        'encounter_ts_wrt_os': encounter_ts_wrt_os,
        'encounter_os_wrt_ts': encounter_os_wrt_ts
    }
    return get_recommendation_from_encounters(encounter_ts_wrt_os, encounter_os_wrt_ts, good_seamanship=good_seamanship, ts_in_TSS=ts_in_TSS, os_in_TSS=os_in_TSS), info