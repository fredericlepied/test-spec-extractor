import { HashRouter, Routes, Route } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { Dashboard } from "./components/dashboard/Dashboard";
import { SimilarityExplorer } from "./components/similarity/SimilarityExplorer";
import { TestCatalog } from "./components/catalog/TestCatalog";
import { ClustersView } from "./components/clusters/ClustersView";
import { SimilarityGraph } from "./components/graph/SimilarityGraph";

export default function App() {
  return (
    <HashRouter>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/similarity" element={<SimilarityExplorer />} />
          <Route path="/clusters" element={<ClustersView />} />
          <Route path="/graph" element={<SimilarityGraph />} />
          <Route path="/catalog" element={<TestCatalog />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
