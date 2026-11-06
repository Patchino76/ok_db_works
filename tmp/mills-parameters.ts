// mills-parameters.ts - Unified parameter definitions and utilities for Mills-AI components
import { create } from "zustand";
import { ModelParameter } from "../types/parameters";

/**
 * Default list of model parameters with Bulgarian names and descriptions
 */

// Icons for parameters
export const parameterIcons: Record<string, string> = {
  Ore: "⛏️",
  WaterMill: "💧",
  WaterZumpf: "🌊",
  MotorAmp: "⚡",
  PressureHC: "📊",
  DensityHC: "🧪",
  PumpRPM: "🔄",
  CirculativeLoad: "🔁",
  Shisti: "🪨",
  Daiki: "🧬",
  Grano: "📏",
  Class_12: "🔢",
  Class_15: "🔢",
  FE: "🔢",
  PSI80: "🎯",
  PSI200: "🎯",
};

// Colors for parameters
export const parameterColors: Record<string, string> = {
  Ore: "amber",
  WaterMill: "blue",
  WaterZumpf: "cyan",
  PressureHC: "red",
  DensityHC: "purple",
  MotorAmp: "yellow",
  Shisti: "green",
  Daiki: "orange",
  PumpRPM: "indigo",
  CirculativeLoad: "teal",
  Grano: "slate",
  Class_12: "rose",
  Class_15: "rose",
  FE: "rose",
  PSI80: "green",
  PSI200: "green",
};

// Helper function to calculate 10% inside ranges
const calculateDefaultRange = (min: number, max: number): [number, number] => {
  const range = max - min;
  const margin = range * 0.1; // 10% margin
  return [min + margin, max - margin];
};

// Unified parameter definitions with Bulgarian translations
export const millsParameters: ModelParameter[] = [
  {
    id: "Ore",
    name: "Разход на руда",
    type: "feature",
    enabled: true,
    filterEnabled: false, // Disabled by default for features
    min: 140,
    max: 220,
    currentMin: calculateDefaultRange(140, 240)[0], // 10% inside
    currentMax: calculateDefaultRange(140, 240)[1], // 10% inside
    unit: "t/h",
    varType: "MV",
    description: "Разход на входяща руда към мелницата",
  },
  {
    id: "WaterMill",
    name: "Вода в мелницата",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 5,
    max: 25,
    currentMin: calculateDefaultRange(5, 25)[0],
    currentMax: calculateDefaultRange(5, 25)[1],
    unit: "m³/h",
    varType: "MV",
    description: "Разход на вода в мелницата",
  },
  {
    id: "WaterZumpf",
    name: "Вода в зумпфа",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 140,
    max: 250,
    currentMin: calculateDefaultRange(140, 250)[0],
    currentMax: calculateDefaultRange(140, 250)[1],
    unit: "m³/h",
    varType: "MV",
    description: "Разход на вода в зумпф",
  },
  {
    id: "PulpHC",
    name: "Пулп в ХЦ",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 400,
    max: 600,
    currentMin: calculateDefaultRange(400, 600)[0],
    currentMax: calculateDefaultRange(400, 600)[1],
    unit: "m³/h",
    varType: "CV",
    description: "Разход на пулп в ХЦ",
  },
  {
    id: "DensityHC",
    name: "Плътност на ХЦ",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 1500,
    max: 1900,
    currentMin: calculateDefaultRange(1500, 1900)[0],
    currentMax: calculateDefaultRange(1500, 1900)[1],
    unit: "kg/m³",
    varType: "CV",
    description: "Плътност на пулп в хидроциклона",
  },
  {
    id: "PressureHC",
    name: "Налягане на ХЦ",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 0.2,
    max: 0.5,
    currentMin: calculateDefaultRange(0.2, 0.5)[0],
    currentMax: calculateDefaultRange(0.2, 0.5)[1],
    unit: "bar",
    varType: "CV",
    description: "Работно налягане в хидроциклона",
  },
  {
    id: "PumpRPM",
    name: "Обороти на помпата",
    type: "feature",
    enabled: false,
    filterEnabled: false,
    min: 0,
    max: 800,
    currentMin: calculateDefaultRange(0, 800)[0],
    currentMax: calculateDefaultRange(0, 800)[1],
    unit: "rev/min",
    varType: "CV",
    description: "Обороти на работната помпа",
  },
  {
    id: "CirculativeLoad",
    name: "Цирк. товар",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 1.0,
    max: 3.0,
    currentMin: calculateDefaultRange(0, 3.0)[0],
    currentMax: calculateDefaultRange(0, 3.0)[1],
    unit: "t/t",
    varType: "CV",
    description: "Циркулационен товар на твърдо вещество в МА",
  },
  {
    id: "MotorAmp",
    name: "Ток на елетродвигателя",
    type: "feature",
    enabled: true,
    filterEnabled: true,
    min: 150,
    max: 250,
    currentMin: calculateDefaultRange(150, 250)[0],
    currentMax: calculateDefaultRange(150, 250)[1],
    unit: "A",
    varType: "DV",
    description: "Консумация на ток от електродвигателя на мелницата",
    hasTrend: true, // Real-time data available from endpoints
    hasSmoothing: true, // Apply moving average smoothing to trend data
  },
  {
    id: "Shisti",
    name: "Шисти",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 0.0,
    max: 100.0,
    currentMin: calculateDefaultRange(0.0, 100.0)[0],
    currentMax: calculateDefaultRange(0.0, 100.0)[1],
    unit: "%",
    varType: "DV",
    description: "Процентно съдържание на шисти в рудата",
  },
  {
    id: "Daiki",
    name: "Дайки",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 0.0,
    max: 100.0,
    currentMin: calculateDefaultRange(0.0, 100.0)[0],
    currentMax: calculateDefaultRange(0.0, 100.0)[1],
    unit: "%",
    varType: "DV",
    description: "Процентно съдържание на дайки в рудата",
  },
  {
    id: "Grano",
    name: "Гранодиорити",
    type: "feature",
    enabled: false,
    filterEnabled: false,
    min: 0.0,
    max: 100.0,
    currentMin: calculateDefaultRange(0.0, 100.0)[0],
    currentMax: calculateDefaultRange(0.0, 100.0)[1],
    unit: "%",
    varType: "DV",
    description: "Процентно съдържание на гранодиорити в рудата",
  },
  {
    id: "Class_12",
    name: "Клас 12",
    type: "feature",
    enabled: false,
    filterEnabled: false,
    min: 0.0,
    max: 5.0,
    currentMin: calculateDefaultRange(0.0, 5.0)[0],
    currentMax: calculateDefaultRange(0.0, 5.0)[1],
    unit: "%",
    varType: "DV",
    description: "Процент материал в клас +12 милиметра",
  },
  {
    id: "Class_15",
    name: "Клас 15",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 0.0,
    max: 2.0,
    currentMin: calculateDefaultRange(0.0, 2.0)[0],
    currentMax: calculateDefaultRange(0.0, 2.0)[1],
    unit: "%",
    varType: "DV",
    description: "Процент материал в клас +15 милиметра",
  },
  {
    id: "FE",
    name: "Желязо",
    type: "feature",
    enabled: true,
    filterEnabled: false,
    min: 2.0,
    max: 5.0,
    currentMin: calculateDefaultRange(0.0, 2.6)[0],
    currentMax: calculateDefaultRange(0.0, 4.6)[1],
    unit: "%",
    varType: "DV",
    description: "Процент съдържание на желязо в пулпа",
    hasTrend: true, // Real-time data available from endpoints
  },
  {
    id: "PSI80",
    name: "Фракция -80 μk",
    type: "target",
    enabled: false,
    filterEnabled: true, // Enabled by default for targets
    min: 40,
    max: 60,
    currentMin: calculateDefaultRange(40.0, 60.0)[0],
    currentMax: calculateDefaultRange(40.0, 60.0)[1],
    unit: "%",
    varType: "TARGET", // Target variable, not MV
    description: "Класификация на размерите на частиците при 80 микрона",
  },
  {
    id: "PSI200",
    name: "Фракция +200 μk",
    type: "target",
    enabled: true,
    filterEnabled: true, // Enabled by default for targets
    min: 16,
    max: 30,
    currentMin: calculateDefaultRange(16.0, 30.0)[0],
    currentMax: calculateDefaultRange(16.0, 30.0)[1],
    unit: "%",
    varType: "TARGET", // Target variable, not MV
    description: "Основна целева стойност - финност на смилане +200 микрона",
  },
];

// Helper functions for parameters
export const getParameterDefaultValues = () => {
  return millsParameters.reduce((acc, param) => {
    acc[param.id] = (param.min + param.max) / 2;
    return acc;
  }, {} as Record<string, number>);
};

export const getParameterBounds = () => {
  return millsParameters.reduce((acc, param) => {
    acc[param.id] = [param.min, param.max];
    return acc;
  }, {} as Record<string, [number, number]>);
};

// Extract just the features or targets
export const getFeatures = () =>
  millsParameters.filter((p) => p.type === "feature");
export const getTargets = () =>
  millsParameters.filter((p) => p.type === "target");

// Helper to map parameters to the XGBoost store format
export const mapToStoreParameters = () => {
  return millsParameters
    .filter((p) => p.type === "feature")
    .map((param) => ({
      id: param.id,
      name: param.name,
      unit: param.unit,
      value: (param.currentMin + param.currentMax) / 2,
      trend: [] as Array<{ timestamp: number; value: number }>,
      color: parameterColors[param.id] || "gray",
      icon: parameterIcons[param.id] || "📊",
    }));
};

// Custom hook to use parameters with state
export const useDefaultParameters = () => {
  return millsParameters;
};
