import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    include: ["vis-network/standalone/esm/vis-network"],
  },
  server: {
    host: "0.0.0.0",
    port: 5173
  }
});
