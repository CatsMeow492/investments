import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TableSortLabel,
  List,
  ListItem,
  Link,
} from '@mui/material';
import {
  PieChart,
  Pie,
  Cell,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { usePortfolio } from '../contexts/PortfolioContext';

const COLORS = [
  '#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', 
  '#82CA9D', '#A4DE6C', '#D0ED57', '#FFC658', '#FF7C43'
];

interface SortConfig {
  key: string;
  direction: 'asc' | 'desc';
}

const Dashboard: React.FC = () => {
  const {
    summary,
    allocation,
    recommendations,
    loading,
    errors,
    lastUpdate,
    nextUpdate
  } = usePortfolio();

  const [sortConfig, setSortConfig] = React.useState<SortConfig>({
    key: 'percentage',
    direction: 'desc'
  });

  const handleSort = (key: string) => {
    setSortConfig(current => ({
      key,
      direction: current.key === key && current.direction === 'desc' ? 'asc' : 'desc'
    }));
  };

  const getSortedHoldings = () => {
    if (!allocation?.by_holding) return [];
    
    return [...allocation.by_holding].sort((a, b) => {
      if (sortConfig.direction === 'asc') {
        return a[sortConfig.key as keyof typeof a] > b[sortConfig.key as keyof typeof b] ? 1 : -1;
      }
      return a[sortConfig.key as keyof typeof a] < b[sortConfig.key as keyof typeof b] ? 1 : -1;
    });
  };

  const typeAllocationData = allocation ? 
    Object.entries(allocation.by_type).map(([name, value]) => ({ 
      name, 
      value: parseFloat(value.toFixed(2))
    })) : [];

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload;
      return (
        <Paper sx={{ p: 1.5 }}>
          <Typography variant="subtitle2">
            {data.fullName || data.name}
          </Typography>
          <Typography variant="body2" color="text.secondary">
            {`${data.value}%`}
          </Typography>
          {data.fullName && (
            <Typography variant="body2" color="text.secondary">
              ${(data.value * (allocation?.total_value || 0) / 100).toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2
              })}
            </Typography>
          )}
        </Paper>
      );
    }
    return null;
  };

  return (
    <Grid container spacing={3}>
      {/* Portfolio Summary */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Typography variant="h5" gutterBottom>
            Portfolio Summary
          </Typography>
          {loading.summary ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : errors.summary ? (
            <Typography color="error" sx={{ p: 2 }}>{errors.summary}</Typography>
          ) : summary && (
            <Grid container spacing={3}>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography color="text.secondary">Total Value</Typography>
                  <Typography variant="h4">
                    ${summary.total_value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography color="text.secondary">Total Gain/Loss</Typography>
                  <Typography 
                    variant="h4" 
                    color={summary.total_gain_loss >= 0 ? 'success.main' : 'error.main'}
                  >
                    {summary.total_gain_loss >= 0 ? '+' : '-'}$
                    {Math.abs(summary.total_gain_loss).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                  </Typography>
                </Box>
              </Grid>
              <Grid item xs={12} sm={4}>
                <Box>
                  <Typography color="text.secondary">Return</Typography>
                  <Typography 
                    variant="h4" 
                    color={summary.gain_loss_percentage >= 0 ? 'success.main' : 'error.main'}
                  >
                    {summary.gain_loss_percentage >= 0 ? '+' : '-'}
                    {Math.abs(summary.gain_loss_percentage).toFixed(2)}%
                  </Typography>
                </Box>
              </Grid>
            </Grid>
          )}
        </Paper>
      </Grid>

      {/* Asset Type Allocation */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3, height: 400 }}>
          <Typography variant="h6" gutterBottom>
            Asset Type Allocation
          </Typography>
          {loading.allocation ? (
            <Box display="flex" justifyContent="center" alignItems="center" height="calc(100% - 32px)">
              <CircularProgress />
            </Box>
          ) : errors.allocation ? (
            <Typography color="error" sx={{ p: 2 }}>{errors.allocation}</Typography>
          ) : allocation && (
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={typeAllocationData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={80}
                  fill="#8884d8"
                  paddingAngle={5}
                  dataKey="value"
                  label={({ name, value }) => `${name} (${value}%)`}
                >
                  {typeAllocationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </Paper>
      </Grid>

      {/* Holdings Table */}
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 3, height: 400, overflow: 'auto' }}>
          <Typography variant="h6" gutterBottom>
            Holdings Breakdown
          </Typography>
          {loading.allocation ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : errors.allocation ? (
            <Typography color="error" sx={{ p: 2 }}>{errors.allocation}</Typography>
          ) : allocation && (
            <TableContainer>
              <Table size="small" stickyHeader>
                <TableHead>
                  <TableRow>
                    <TableCell>
                      <TableSortLabel
                        active={sortConfig.key === 'symbol'}
                        direction={sortConfig.direction}
                        onClick={() => handleSort('symbol')}
                      >
                        Symbol
                      </TableSortLabel>
                    </TableCell>
                    <TableCell>
                      <TableSortLabel
                        active={sortConfig.key === 'name'}
                        direction={sortConfig.direction}
                        onClick={() => handleSort('name')}
                      >
                        Name
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={sortConfig.key === 'value'}
                        direction={sortConfig.direction}
                        onClick={() => handleSort('value')}
                      >
                        Value
                      </TableSortLabel>
                    </TableCell>
                    <TableCell align="right">
                      <TableSortLabel
                        active={sortConfig.key === 'percentage'}
                        direction={sortConfig.direction}
                        onClick={() => handleSort('percentage')}
                      >
                        Weight
                      </TableSortLabel>
                    </TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {getSortedHoldings().map((holding) => (
                    <TableRow 
                      key={holding.symbol}
                      sx={{ '&:last-child td, &:last-child th': { border: 0 } }}
                    >
                      <TableCell component="th" scope="row">
                        <Typography variant="body2" fontWeight="bold">
                          {holding.symbol}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2" noWrap>
                          {holding.name}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">
                          ${holding.value.toLocaleString(undefined, {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2
                          })}
                        </Typography>
                      </TableCell>
                      <TableCell align="right">
                        <Typography variant="body2">
                          {holding.percentage.toFixed(2)}%
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Grid>

      {/* AI Recommendations */}
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
            <Typography variant="h6">
              AI Insights & Recommendations
            </Typography>
          </Box>
          
          {errors.recommendations ? (
            <Box p={2} bgcolor="error.light" borderRadius={1}>
              <Typography color="error" variant="body2">
                {errors.recommendations}
              </Typography>
              <Typography variant="caption" color="error.dark" sx={{ mt: 1, display: 'block' }}>
                This could be due to API rate limits or server processing. Please try again later.
              </Typography>
            </Box>
          ) : (
            <TableContainer>
              <Table size="medium">
                <TableHead>
                  <TableRow>
                    <TableCell>Asset</TableCell>
                    <TableCell>Current Value</TableCell>
                    <TableCell>Analysis</TableCell>
                    <TableCell>Recent News</TableCell>
                    <TableCell>Last Updated</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {allocation?.by_holding.map((holding) => (
                    <TableRow key={holding.symbol}>
                      <TableCell>
                        <Typography variant="body2" fontWeight="bold">
                          {holding.symbol}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {holding.name}
                        </Typography>
                      </TableCell>
                      <TableCell>
                        <Typography variant="body2">
                          ${holding.value.toLocaleString(undefined, {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2
                          })}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {holding.percentage.toFixed(2)}% of portfolio
                        </Typography>
                      </TableCell>
                      <TableCell>
                        {loading.recommendations ? (
                          <Box display="flex" alignItems="center" gap={1}>
                            <CircularProgress size={16} />
                            <Typography variant="caption" color="text.secondary">
                              Generating insights...
                            </Typography>
                          </Box>
                        ) : (
                          recommendations.find(rec => rec.symbol === holding.symbol)?.analysis || (
                            <Typography variant="body2" color="text.secondary">
                              Analysis pending...
                            </Typography>
                          )
                        )}
                      </TableCell>
                      <TableCell>
                        {loading.recommendations ? (
                          <Box display="flex" alignItems="center" gap={1}>
                            <CircularProgress size={16} />
                            <Typography variant="caption" color="text.secondary">
                              Fetching news...
                            </Typography>
                          </Box>
                        ) : (
                          <List dense>
                            {recommendations.find(rec => rec.symbol === holding.symbol)?.news?.map((news, index) => (
                              <ListItem key={index} disablePadding>
                                <Link
                                  href={news.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  color="inherit"
                                  underline="hover"
                                >
                                  <Typography variant="caption">
                                    {news.title}
                                  </Typography>
                                </Link>
                              </ListItem>
                            )) || (
                              <Typography variant="caption" color="text.secondary">
                                News pending...
                              </Typography>
                            )}
                          </List>
                        )}
                      </TableCell>
                      <TableCell>
                        <Typography variant="caption" color="text.secondary">
                          {recommendations.find(rec => rec.symbol === holding.symbol)?.last_updated
                            ? new Date(recommendations.find(rec => rec.symbol === holding.symbol)!.last_updated).toLocaleString()
                            : 'Pending update...'}
                        </Typography>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          )}
        </Paper>
      </Grid>

      {/* Last Updated */}
      {lastUpdate && (
        <Grid item xs={12}>
          <Typography variant="caption" color="text.secondary">
            Last updated: {lastUpdate.toLocaleString()}
            {nextUpdate && ` (Next update: ${nextUpdate.toLocaleString()})`}
          </Typography>
        </Grid>
      )}
    </Grid>
  );
};

export default Dashboard; 