import react from '@vitejs/plugin-react';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  // Reads frontend/.env (all variables, not only VITE_*) for dev-server settings.
  const env = loadEnv(mode, process.cwd(), '');

  return {
    plugins: [react()],
    server: {
      host: env.DEV_SERVER_HOST || '127.0.0.1',
      port: Number(env.DEV_SERVER_PORT || 5173),
      strictPort: true,
      // The browser calls /api on the Vite server; Vite forwards it to FastAPI.
      // No CORS issues in development and no API URL hard-coded in the React code.
      proxy: {
        '/api': { target: env.API_PROXY_TARGET || 'http://127.0.0.1:8000', changeOrigin: true },
      },
    },
  };
});
