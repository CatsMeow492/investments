import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, Container } from '@mui/material';
import { PortfolioProvider } from './contexts/PortfolioContext';
import Dashboard from './pages/Dashboard';
import Portfolio from './pages/Portfolio';
import Research from './pages/Research';
import Navbar from './components/Navbar';

const App: React.FC = () => {
  return (
    <PortfolioProvider>
      <Box sx={{ display: 'flex', flexDirection: 'column', minHeight: '100vh' }}>
        <Navbar />
        <Container component="main" sx={{ mt: 4, mb: 4, flex: 1 }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/research" element={<Research />} />
          </Routes>
        </Container>
      </Box>
    </PortfolioProvider>
  );
};

export default App;
