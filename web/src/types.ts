export interface TestRecord {
  id: string;
  desc: string;
  repo: string;
  language: "go" | "python";
  filePath: string;
  lineNumber: number | null;
  testId: string | null;
  polarionUrl: string | null;
  sourceUrl: string | null;
  steps: string[];
  validations: string[];
  k8sResources: string[];
  labels: string[];
  prepSteps: string[];
  skipConditions: string[];
  cleanupSteps: string[];
}

export interface SimilarityMatch {
  id: number;
  queryDesc: string;
  queryFile: string;
  queryRepo: string;
  querySourceUrl: string | null;
  matchedDesc: string;
  matchedFile: string;
  matchedRepo: string;
  matchedSourceUrl: string | null;
  semanticSimilarity: number;
  contextSimilarity: number;
  isCrossLanguage: boolean;
  matchType: "go-go" | "py-py" | "cross";
  sharedLabels: string[];
  querySteps: string[];
  queryValidations: string[];
  queryK8sResources: string[];
  matchedSteps: string[];
  matchedValidations: string[];
  matchedK8sResources: string[];
}

export interface DashboardStats {
  totalTests: number;
  goTests: number;
  pyTests: number;
  totalMatches: number;
  crossLanguageMatches: number;
  avgSimilarity: number;
  repos: { name: string; count: number; language: string }[];
  scoreDistribution: { bucket: string; count: number }[];
  k8sResources: { name: string; count: number }[];
  commonK8sResources: string[];
  testIdCoverage: { withId: number; withoutId: number };
  matchTypes: { goGo: number; pyPy: number; cross: number };
  repoOverlap: { repo: string; total: number; crossRepo: number; crossRepoPct: number; internal: number; internalPct: number }[];
  heatmap: HeatmapData;
}

export interface HeatmapData {
  repos: string[];
  resources: string[];
  cells: { repo: string; resource: string; count: number }[];
}

export interface ClusterTest {
  desc: string;
  repo: string;
  file: string;
  sourceUrl: string | null;
  language: "go" | "python";
  testId: string | null;
  x: number;
  y: number;
  degree: number;
  colorIndex: number;
}

export interface ClusterEdge {
  sourceIdx: number;
  targetIdx: number;
  score: number;
}

export interface DuplicateCluster {
  id: number;
  size: number;
  repos: string[];
  isCrossRepo: boolean;
  maxScore: number;
  avgScore: number;
  tests: ClusterTest[];
  edges: ClusterEdge[];
}
