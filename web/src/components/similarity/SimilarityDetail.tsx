import type { SimilarityMatch, TestRecord } from "../../types";
import { ScoreBadge } from "./ScoreBadge";

interface Props {
  match: SimilarityMatch;
  queryTest: TestRecord | null;
  matchedTest: TestRecord | null;
}

function Section({ title, items }: { title: string; items: string[] }) {
  if (items.length === 0) return null;
  return (
    <div>
      <div className="text-xs font-medium text-gray-400 mb-1">{title}</div>
      <ul className="list-disc list-inside text-xs text-gray-600 dark:text-gray-300 space-y-0.5">
        {items.map((item, i) => <li key={i}>{item}</li>)}
      </ul>
    </div>
  );
}

function TestSide({ label, desc, file, sourceUrl, test }: {
  label: string;
  desc: string;
  file: string;
  sourceUrl: string | null;
  test: TestRecord | null;
}) {
  const steps = test?.steps ?? [];
  const validations = test?.validations ?? [];
  const k8sResources = test?.k8sResources ?? [];
  const prepSteps = test?.prepSteps ?? [];
  const skipConditions = test?.skipConditions ?? [];
  const cleanupSteps = test?.cleanupSteps ?? [];
  const polarionUrl = test?.polarionUrl;
  const testId = test?.testId;

  return (
    <div className="flex-1 min-w-0">
      <div className="text-xs font-medium text-gray-400 dark:text-gray-500 mb-1">{label}</div>
      <div className="text-sm font-medium mb-1">{desc}</div>
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <span className="text-xs text-gray-500 dark:text-gray-400 truncate">
          {sourceUrl ? (
            <a href={sourceUrl} target="_blank" rel="noopener noreferrer" className="text-blue-600 dark:text-blue-400 hover:underline">{file}</a>
          ) : file}
        </span>
        {polarionUrl && testId && (
          <a href={polarionUrl} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-600 dark:text-blue-400 hover:underline">
            OCP-{testId}
          </a>
        )}
      </div>

      {k8sResources.length > 0 && (
        <div className="mb-3">
          <div className="flex flex-wrap gap-1">
            {k8sResources.map((r) => (
              <span key={r} className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-xs">{r}</span>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-2">
        <Section title="Skip Conditions" items={skipConditions} />
        <Section title="Preparation" items={prepSteps} />
        <Section title="Steps" items={steps} />
        <Section title="Validations" items={validations} />
        <Section title="Cleanup" items={cleanupSteps} />
      </div>
    </div>
  );
}

export function SimilarityDetail({ match, queryTest, matchedTest }: Props) {
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
      <div className="flex items-center gap-3 mb-4">
        <ScoreBadge score={match.semanticSimilarity} />
        <span className="text-xs text-gray-400">{match.matchType}</span>
        {match.sharedLabels.length > 0 && (
          <span className="text-xs text-gray-400">Shared: {match.sharedLabels.join(", ")}</span>
        )}
      </div>
      <div className="flex gap-6">
        <TestSide
          label="Test A"
          desc={match.queryDesc}
          file={match.queryFile}
          sourceUrl={match.querySourceUrl}
          test={queryTest}
        />
        <div className="w-px bg-gray-200 dark:bg-gray-700 shrink-0" />
        <TestSide
          label="Test B"
          desc={match.matchedDesc}
          file={match.matchedFile}
          sourceUrl={match.matchedSourceUrl}
          test={matchedTest}
        />
      </div>
    </div>
  );
}
