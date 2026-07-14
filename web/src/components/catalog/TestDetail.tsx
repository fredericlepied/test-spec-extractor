import type { TestRecord } from "../../types";

interface Props {
  test: TestRecord;
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

export function TestDetail({ test }: Props) {
  return (
    <div className="bg-gray-50 dark:bg-gray-800/50 p-4 space-y-3 text-sm">
      <div className="flex flex-wrap gap-4">
        <div>
          <span className="text-xs text-gray-400">File: </span>
          {test.sourceUrl ? (
            <a href={test.sourceUrl} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-600 dark:text-blue-400 hover:underline">
              {test.filePath}:{test.lineNumber}
            </a>
          ) : (
            <span className="text-xs text-gray-600 dark:text-gray-300">{test.filePath}</span>
          )}
        </div>
        {test.polarionUrl && (
          <div>
            <span className="text-xs text-gray-400">Polarion: </span>
            <a href={test.polarionUrl} target="_blank" rel="noopener noreferrer" className="text-xs text-blue-600 dark:text-blue-400 hover:underline">
              OCP-{test.testId}
            </a>
          </div>
        )}
      </div>

      {test.k8sResources.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {test.k8sResources.map((r) => (
            <span key={r} className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-xs">{r}</span>
          ))}
        </div>
      )}

      {test.labels.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {test.labels.map((l) => (
            <span key={l} className="px-1.5 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded text-xs">{l}</span>
          ))}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <Section title="Skip Conditions" items={test.skipConditions} />
        <Section title="Preparation" items={test.prepSteps} />
        <Section title="Steps" items={test.steps} />
        <Section title="Validations" items={test.validations} />
        <Section title="Cleanup" items={test.cleanupSteps} />
      </div>
    </div>
  );
}
