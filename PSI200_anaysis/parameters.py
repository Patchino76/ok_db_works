"""
Parameter definitions for mill analysis with Bulgarian translations and units.
This file contains parameter metadata for visualization and analysis.
"""

# Parameter dictionary with Bulgarian names and units
mill_parameters = [
    {
        "id": "Ore",
        "name": "Разход на руда",
        "unit": "t/h",
        "description": "Разход на входяща руда към мелницата"
    },
    {
        "id": "WaterMill",
        "name": "Вода в мелницата",
        "unit": "m³/h",
        "description": "Разход на вода в мелницата"
    },
    {
        "id": "WaterZumpf",
        "name": "Вода в зумпфа",
        "unit": "m³/h",
        "description": "Разход на вода в зумпф"
    },
    {
        "id": "PulpHC",
        "name": "Пулп в ХЦ",
        "unit": "m³/h",
        "description": "Разход на пулп в ХЦ"
    },
    {
        "id": "MotorAmp",
        "name": "Ток на елетродвигателя",
        "unit": "A",
        "description": "Консумация на ток от електродвигателя на мелницата"
    },
    {
        "id": "DensityHC",
        "name": "Плътност на ХЦ",
        "unit": "g/L",
        "description": "Плътност на пулп в хидроциклона"
    },
    {
        "id": "PressureHC",
        "name": "Налягане на ХЦ",
        "unit": "bar",
        "description": "Работно налягане в хидроциклона"
    },
    {
        "id": "PumpRPM",
        "name": "Обороти на помпата",
        "unit": "rev/min",
        "description": "Обороти на работната помпа"
    },
    {
        "id": "Shisti",
        "name": "Шисти",
        "unit": "%",
        "description": "Процентно съдържание на шисти в рудата"
    },
    {
        "id": "Daiki",
        "name": "Дайки",
        "unit": "%",
        "description": "Процентно съдържание на дайки в рудата"
    },
    {
        "id": "Grano",
        "name": "Гранодиорити",
        "unit": "%",
        "description": "Процентно съдържание на гранодиорити в рудата"
    },
    {
        "id": "Class_12",
        "name": "Клас 12",
        "unit": "%",
        "description": "Процент материал в клас +12 милиметра"
    },
    {
        "id": "Class_15",
        "name": "Клас 15",
        "unit": "%",
        "description": "Процент материал в клас +15 милиметра"
    },
    {
        "id": "PSI80",
        "name": "Фракция -80 μk",
        "unit": "μk",
        "description": "Основна целева стойност - финност на смилане -80 микрона"
    },
    {
        "id": "PSI200",
        "name": "Фракция +200 μk",
        "unit": "μk",
        "description": "Основна целева стойност - финност на смилане +200 микрона"
    }
]

# Parameter icons
parameter_icons = {
    "Ore": "⛏️",
    "WaterMill": "💧",
    "WaterZumpf": "🌊",
    "PressureHC": "📊",
    "DensityHC": "🧪",
    "MotorAmp": "⚡",
    "Shisti": "🪨",
    "Daiki": "🧬",
    "PumpRPM": "🔄",
    "Grano": "📏",
    "Class_12": "🔢",
    "PSI80": "🎯",
    "PSI200": "🎯"
}

# Parameter colors
parameter_colors = {
    "Ore": "amber",
    "WaterMill": "blue",
    "WaterZumpf": "cyan",
    "PressureHC": "red",
    "DensityHC": "purple",
    "MotorAmp": "yellow",
    "Shisti": "green",
    "Daiki": "orange",
    "PumpRPM": "indigo",
    "Grano": "slate",
    "Class_12": "rose",
    "PSI80": "green",
    "PSI200": "green"
}

# Helper function to get parameter info
def get_parameter_info(parameter_id):
    """
    Get parameter information by ID
    
    Args:
        parameter_id (str): The parameter ID to look up
        
    Returns:
        dict: Parameter information including name, unit, etc. or None if not found
    """
    for param in mill_parameters:
        if param["id"] == parameter_id:
            return param
    return None

# Helper function to get formatted parameter name with unit
def get_formatted_name(parameter_id):
    """
    Get formatted parameter name with unit for plots
    
    Args:
        parameter_id (str): The parameter ID to look up
        
    Returns:
        str: Formatted name with unit or the original ID if not found
    """
    param = get_parameter_info(parameter_id)
    if param:
        return f"{param['name']} ({param['unit']})"
    return parameter_id
