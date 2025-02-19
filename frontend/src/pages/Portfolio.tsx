import React, { useState, useEffect } from 'react';
import axios from 'axios';
import {
  Grid,
  Paper,
  Typography,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  MenuItem,
  Box,
  CircularProgress,
} from '@mui/material';
import { Add as AddIcon, Edit as EditIcon, Delete as DeleteIcon } from '@mui/icons-material';

interface Asset {
  symbol: string;
  name: string;
  type: string;
  quantity: number;
  current_price: number;
  purchase_price: number;
  current_value: number;
  gain_loss: number;
  gain_loss_percentage: number;
  last_updated: string;
  needs_update: boolean;
}

interface PortfolioResponse {
  assets: Asset[];
  update_status: {
    total_assets: number;
    assets_needing_update: number;
    next_update_attempt: string | null;
  };
}

const Portfolio: React.FC = () => {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [openDialog, setOpenDialog] = useState(false);
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [nextUpdate, setNextUpdate] = useState<Date | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshError, setRefreshError] = useState<string | null>(null);
  const [updateStatus, setUpdateStatus] = useState<{
    total_assets: number;
    assets_needing_update: number;
    next_update_attempt: string | null;
  } | null>(null);

  const fetchAssets = async () => {
    try {
      // Only show loading on initial fetch
      if (!assets.length) {
        setLoading(true);
      }
      const response = await axios.get<PortfolioResponse>('http://localhost:8000/api/portfolio/assets');
      
      // Create a Map to ensure uniqueness by symbol
      const uniqueAssets = new Map(
        response.data.assets.map(asset => [asset.symbol, asset])
      );
      
      setAssets(Array.from(uniqueAssets.values()));
      setUpdateStatus(response.data.update_status);
      setLastUpdate(new Date());
      
      if (response.data.update_status.next_update_attempt) {
        setNextUpdate(new Date(response.data.update_status.next_update_attempt));
      }
      setError(null); // Clear any previous errors
    } catch (err) {
      setError('Failed to fetch portfolio assets');
      console.error('Error fetching assets:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    try {
      setRefreshing(true);
      setRefreshError(null);
      const response = await axios.get('http://localhost:8000/api/portfolio/update-prices');
      
      // Handle failed updates
      if (response.data.failed_assets && response.data.failed_assets.length > 0) {
        const failedSymbols = response.data.failed_assets.map((asset: any) => 
          `${asset.symbol} (${asset.reason})`
        ).join(', ');
        setRefreshError(`Failed to update some prices: ${failedSymbols}`);
      }

      // Set next update time
      if (response.data.cache_info && response.data.cache_info.next_update) {
        setNextUpdate(new Date(response.data.cache_info.next_update));
      }
      
      await fetchAssets();
    } catch (err) {
      console.error('Error refreshing prices:', err);
      setRefreshError('Failed to refresh prices. Please try again later.');
    } finally {
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAssets();
    // Refresh data every minute if there are assets needing updates
    const interval = setInterval(() => {
      if (updateStatus && updateStatus.assets_needing_update > 0) {
        fetchAssets();
      }
    }, 60000);
    return () => clearInterval(interval);
  }, [updateStatus?.assets_needing_update]);

  const handleAddClick = () => {
    setSelectedAsset(null);
    setOpenDialog(true);
  };

  const handleEditClick = (asset: Asset) => {
    setSelectedAsset(asset);
    setOpenDialog(true);
  };

  const handleDeleteClick = (symbol: string) => {
    // TODO: Implement delete functionality with backend API
    setAssets(assets.filter(asset => asset.symbol !== symbol));
  };

  const handleCloseDialog = () => {
    setOpenDialog(false);
    setSelectedAsset(null);
  };

  // Only show loading screen on initial load
  if (loading && !assets.length) {
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

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 3 }}>
          <Grid container alignItems="center" justifyContent="space-between" mb={3}>
            <Box>
              <Typography variant="h5">My Investments</Typography>
              {lastUpdate && (
                <Typography variant="caption" color="text.secondary" display="block">
                  Last updated: {lastUpdate.toLocaleTimeString()}
                </Typography>
              )}
              {updateStatus && updateStatus.assets_needing_update > 0 && (
                <Typography variant="caption" color="warning.main" display="block">
                  {updateStatus.assets_needing_update} assets waiting for price updates
                  {nextUpdate && ` (next attempt: ${nextUpdate.toLocaleTimeString()})`}
                </Typography>
              )}
              {refreshError && (
                <Typography variant="caption" color="error" display="block">
                  {refreshError}
                </Typography>
              )}
            </Box>
            <Box sx={{ display: 'flex', gap: 2 }}>
              <Button
                variant="outlined"
                onClick={handleRefresh}
                disabled={refreshing || (nextUpdate ? new Date() < nextUpdate : false)}
                startIcon={<CircularProgress size={16} sx={{ display: refreshing ? 'inline-flex' : 'none' }} />}
              >
                {nextUpdate && new Date() < nextUpdate 
                  ? `Update Available in ${Math.ceil((nextUpdate.getTime() - new Date().getTime()) / (1000 * 60))} minutes`
                  : 'Refresh Prices'
                }
              </Button>
              <Button
                variant="contained"
                startIcon={<AddIcon />}
                onClick={handleAddClick}
              >
                Add Investment
              </Button>
            </Box>
          </Grid>

          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Symbol</TableCell>
                  <TableCell>Name</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell align="right">Quantity</TableCell>
                  <TableCell align="right">Purchase Price</TableCell>
                  <TableCell align="right">Current Price</TableCell>
                  <TableCell align="right">Current Value</TableCell>
                  <TableCell align="right">Gain/Loss</TableCell>
                  <TableCell align="right">Gain/Loss %</TableCell>
                  <TableCell align="center">Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {assets.map((asset) => (
                  <TableRow 
                    key={asset.symbol}
                    sx={asset.needs_update ? { backgroundColor: 'action.hover' } : undefined}
                  >
                    <TableCell>{asset.symbol}</TableCell>
                    <TableCell>{asset.name}</TableCell>
                    <TableCell>{asset.type}</TableCell>
                    <TableCell align="right">{asset.quantity.toFixed(4)}</TableCell>
                    <TableCell align="right">${asset.purchase_price.toFixed(2)}</TableCell>
                    <TableCell align="right">
                      ${asset.current_price.toFixed(2)}
                      <Typography 
                        variant="caption" 
                        display="block" 
                        color={asset.needs_update ? "warning.main" : "text.secondary"}
                      >
                        {asset.needs_update 
                          ? "Update pending..."
                          : new Date(asset.last_updated).toLocaleTimeString()
                        }
                      </Typography>
                    </TableCell>
                    <TableCell align="right">${asset.current_value.toFixed(2)}</TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: asset.gain_loss >= 0 ? 'success.main' : 'error.main',
                      }}
                    >
                      ${Math.abs(asset.gain_loss).toFixed(2)}
                      {asset.gain_loss >= 0 ? ' +' : ' -'}
                    </TableCell>
                    <TableCell
                      align="right"
                      sx={{
                        color: asset.gain_loss_percentage >= 0 ? 'success.main' : 'error.main',
                      }}
                    >
                      {asset.gain_loss_percentage >= 0 ? '+' : '-'}
                      {Math.abs(asset.gain_loss_percentage).toFixed(2)}%
                    </TableCell>
                    <TableCell align="center">
                      <IconButton onClick={() => handleEditClick(asset)} size="small">
                        <EditIcon />
                      </IconButton>
                      <IconButton onClick={() => handleDeleteClick(asset.symbol)} size="small">
                        <DeleteIcon />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      </Grid>

      <Dialog open={openDialog} onClose={handleCloseDialog}>
        <DialogTitle>
          {selectedAsset ? 'Edit Investment' : 'Add Investment'}
        </DialogTitle>
        <DialogContent>
          <Grid container spacing={2} sx={{ mt: 1 }}>
            <Grid item xs={12}>
              <TextField
                label="Symbol"
                fullWidth
                defaultValue={selectedAsset?.symbol}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Name"
                fullWidth
                defaultValue={selectedAsset?.name}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                select
                label="Type"
                fullWidth
                defaultValue={selectedAsset?.type || 'stock'}
              >
                <MenuItem value="stock">Stock</MenuItem>
                <MenuItem value="crypto">Cryptocurrency</MenuItem>
              </TextField>
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Quantity"
                type="number"
                fullWidth
                defaultValue={selectedAsset?.quantity}
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                label="Purchase Price"
                type="number"
                fullWidth
                defaultValue={selectedAsset?.purchase_price}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={handleCloseDialog}>Cancel</Button>
          <Button variant="contained" onClick={handleCloseDialog}>
            {selectedAsset ? 'Save' : 'Add'}
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
};

export default Portfolio; 