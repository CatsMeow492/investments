import React, { useEffect, useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import axios from 'axios';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

interface PortfolioSummary {
  total_value: number;
  total_gain_loss: number;
  gain_loss_percentage: number;
  last_updated: string;
}

interface AssetAllocation {
  [key: string]: number;
}

const Dashboard: React.FC = () => {
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [allocation, setAllocation] = useState<AssetAllocation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        const [summaryRes, allocationRes] = await Promise.all([
          axios.get('http://localhost:8000/api/portfolio/summary'),
          axios.get('http://localhost:8000/api/portfolio/allocation')
        ]);

        setSummary(summaryRes.data);
        setAllocation(allocationRes.data);
      } catch (err) {
        setError('Failed to fetch portfolio data');
        console.error('Error fetching portfolio data:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
    // Refresh data every minute
    const interval = setInterval(fetchData, 60000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="60vh">
        <Typography color="error">{error}</Typography>
      </Box>
    );
  }

  const allocationData = allocation ? 
    Object.entries(allocation).map(([name, value]) => ({ name, value })) : [];

  return (
    <Grid container spacing={3}>
      {/* Portfolio Summary */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>
            Portfolio Summary
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={4}>
              <Box>
                <Typography color="text.secondary">Total Value</Typography>
                <Typography variant="h4">
                  ${(summary?.total_value || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box>
                <Typography color="text.secondary">Total Gain/Loss</Typography>
                <Typography 
                  variant="h4" 
                  color={(summary?.total_gain_loss || 0) >= 0 ? 'success.main' : 'error.main'}
                >
                  {(summary?.total_gain_loss || 0) >= 0 ? '+' : '-'}$
                  {Math.abs(summary?.total_gain_loss || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </Typography>
              </Box>
            </Grid>
            <Grid item xs={12} sm={4}>
              <Box>
                <Typography color="text.secondary">Return</Typography>
                <Typography 
                  variant="h4" 
                  color={(summary?.gain_loss_percentage || 0) >= 0 ? 'success.main' : 'error.main'}
                >
                  {(summary?.gain_loss_percentage || 0) >= 0 ? '+' : '-'}
                  {Math.abs(summary?.gain_loss_percentage || 0).toFixed(2)}%
                </Typography>
              </Box>
            </Grid>
          </Grid>
        </Paper>
      </Grid>

      {/* Asset Allocation */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3, height: 400 }}>
          <Typography variant="h6" gutterBottom>
            Asset Allocation
          </Typography>
          <ResponsiveContainer width="100%" height="100%">
            <PieChart>
              <Pie
                data={allocationData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={80}
                fill="#8884d8"
                paddingAngle={5}
                dataKey="value"
                label={({ name, value }) => `${name} (${value.toFixed(1)}%)`}
              >
                {allocationData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      {/* Last Updated */}
      <Grid item xs={12}>
        <Typography variant="caption" color="text.secondary">
          Last updated: {new Date(summary?.last_updated || '').toLocaleString()}
        </Typography>
      </Grid>
    </Grid>
  );
};

export default Dashboard; 