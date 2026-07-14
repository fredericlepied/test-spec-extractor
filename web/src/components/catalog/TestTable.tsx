import { useState } from "react";
import type { TestRecord } from "../../types";
import { TestDetail } from "./TestDetail";

interface Props {
  tests: TestRecord[];
  page: number;
  pageSize: number;
  onPageChange: (p: number) => void;
}

export function TestTable({ tests, page, pageSize, onPageChange }: Props) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const totalPages = Math.ceil(tests.length / pageSize);
  const pageTests = tests.slice(page * pageSize, (page + 1) * pageSize);

  return (
    <div>
      <div className="overflow-auto rounded-lg border border-gray-200 dark:border-gray-700">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
            <tr>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Description</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Repo</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Lang</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Resources</th>
              <th className="px-3 py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">Test ID</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
            {pageTests.map((t) => (
              <tr key={t.id} className="group">
                <td colSpan={5} className="p-0">
                  <div
                    onClick={() => setExpandedId(expandedId === t.id ? null : t.id)}
                    className="flex cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
                  >
                    <div className="px-3 py-2 flex-1 min-w-0 truncate">{t.desc}</div>
                    <div className="px-3 py-2 w-36 shrink-0 text-xs text-gray-500 truncate">{t.repo}</div>
                    <div className="px-3 py-2 w-16 shrink-0 text-xs text-gray-400">{t.language === "go" ? "Go" : "Py"}</div>
                    <div className="px-3 py-2 w-40 shrink-0">
                      <div className="flex flex-wrap gap-0.5">
                        {t.k8sResources.slice(0, 3).map((r) => (
                          <span key={r} className="px-1 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 rounded text-[10px]">{r}</span>
                        ))}
                        {t.k8sResources.length > 3 && <span className="text-[10px] text-gray-400">+{t.k8sResources.length - 3}</span>}
                      </div>
                    </div>
                    <div className="px-3 py-2 w-24 shrink-0 text-xs text-gray-400">{t.testId ? `OCP-${t.testId}` : ""}</div>
                  </div>
                  {expandedId === t.id && <TestDetail test={t} />}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
        <span>{tests.length} tests</span>
        <div className="flex gap-2">
          <button disabled={page === 0} onClick={() => onPageChange(page - 1)} className="px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 disabled:opacity-30">Prev</button>
          <span className="px-2 py-1">{page + 1} / {totalPages || 1}</span>
          <button disabled={page >= totalPages - 1} onClick={() => onPageChange(page + 1)} className="px-2 py-1 rounded bg-gray-100 dark:bg-gray-700 disabled:opacity-30">Next</button>
        </div>
      </div>
    </div>
  );
}
