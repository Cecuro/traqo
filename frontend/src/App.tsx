import { HashRouter, Routes, Route } from "react-router-dom";
import { AppProvider, ThemeProvider } from "./context";
import { BrowseView } from "./components/browse/BrowseView";
import { TraceDetailView } from "./components/detail/TraceDetailView";
import { Header } from "./components/Header";
import { ErrorBanner } from "./components/ErrorBanner";

export default function App() {
  return (
    <ThemeProvider>
      <AppProvider>
        <HashRouter>
          <Header />
          <ErrorBanner />
          <main className="flex-1 overflow-auto relative">
            <Routes>
              <Route path="/" element={<BrowseView />} />
              <Route path="/dir/*" element={<BrowseView />} />
              <Route path="/trace/:file" element={<TraceDetailView />} />
              <Route
                path="/trace/:file/:spanId"
                element={<TraceDetailView />}
              />
            </Routes>
          </main>
        </HashRouter>
      </AppProvider>
    </ThemeProvider>
  );
}
