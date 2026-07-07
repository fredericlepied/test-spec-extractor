import { useNavigate } from "react-router-dom";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";

interface Props {
  data: { bucket: string; count: number }[];
}

export function ScoreHistogram({ data }: Props) {
  const navigate = useNavigate();

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-5">
      <h3 className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-4">Similarity Score Distribution</h3>
      <ResponsiveContainer width="100%" height={250}>
        <BarChart
          data={data}
          margin={{ bottom: 20 }}
          onClick={(state) => {
            if (state?.activeLabel) {
              const min = String(state.activeLabel).split("-")[0];
              navigate(`/similarity?min=${min}`);
            }
          }}
          style={{ cursor: "pointer" }}
        >
          <XAxis dataKey="bucket" fontSize={11} tick={{ fill: "#9ca3af" }} />
          <YAxis fontSize={11} tick={{ fill: "#9ca3af" }} />
          <Tooltip
            contentStyle={{ backgroundColor: "var(--color-tooltip-bg)", border: "1px solid var(--color-tooltip-border)", borderRadius: 8, color: "var(--color-tooltip-text)" }}
            labelStyle={{ color: "var(--color-tooltip-text)" }}
            itemStyle={{ color: "var(--color-tooltip-text)" }}
          />
          <Bar dataKey="count" fill="#6366f1" radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
