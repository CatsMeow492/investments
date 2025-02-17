import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import axios from 'axios';

// Types
interface Asset {
  symbol: string;
  name: string;
  type: string;
  quantity: number;
  current_price: number;
  current_value: number;
  purchase_price: number;
  gain_loss: number;
  gain_loss_percentage: number;
  last_updated: string;
  needs_update: boolean;
}

interface PortfolioSummary {
  total_value: number;
  total_gain_loss: number;
  gain_loss_percentage: number;
  last_updated: string;
}

interface AssetAllocation {
  total_value: number;
  by_type: { [key: string]: number };
  by_holding: Array<{
    symbol: string;
    name: string;
    value: number;
    percentage: number;
  }>;
}

interface Recommendation {
  symbol: string;
  name: string;
  value: number;
  analysis: string;
  news: Array<{
    title: string;
    url: string;
    published_at: string;
  }>;
  last_updated: string;
}

interface CacheItem<T> {
  data: T;
  timestamp: number;
  expiry: number;
}

interface PortfolioContextType {
  assets: Asset[];
  summary: PortfolioSummary | null;
  allocation: AssetAllocation | null;
  recommendations: Recommendation[];
  loading: {
    assets: boolean;
    summary: boolean;
    allocation: boolean;
    recommendations: boolean;
  };
  errors: {
    assets: string | null;
    summary: string | null;
    allocation: string | null;
    recommendations: string | null;
  };
  refreshAssets: () => Promise<void>;
  refreshPrices: () => Promise<void>;
  refreshRecommendations: () => Promise<void>;
  lastUpdate: Date | null;
  nextUpdate: Date | null;
}

const CACHE_KEYS = {
  ASSETS: 'portfolio_assets',
  SUMMARY: 'portfolio_summary',
  ALLOCATION: 'portfolio_allocation',
  RECOMMENDATIONS: 'portfolio_recommendations'
} as const;

const CACHE_EXPIRY = {
  ASSETS: 5 * 60 * 1000, // 5 minutes
  SUMMARY: 5 * 60 * 1000, // 5 minutes
  ALLOCATION: 24 * 60 * 60 * 1000, // 24 hours
  RECOMMENDATIONS: 12 * 60 * 60 * 1000 // 12 hours
} as const;

const PortfolioContext = createContext<PortfolioContextType | undefined>(undefined);

const getCachedData = <T,>(key: string): T | null => {
  const cached = localStorage.getItem(key);
  if (!cached) return null;

  try {
    const item: CacheItem<T> = JSON.parse(cached);
    if (Date.now() - item.timestamp > item.expiry) {
      localStorage.removeItem(key);
      return null;
    }
    return item.data;
  } catch {
    localStorage.removeItem(key);
    return null;
  }
};

const setCachedData = <T,>(key: string, data: T, expiry: number) => {
  const item: CacheItem<T> = {
    data,
    timestamp: Date.now(),
    expiry
  };
  localStorage.setItem(key, JSON.stringify(item));
};

export const PortfolioProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  // Clear localStorage cache on mount
  useEffect(() => {
    Object.values(CACHE_KEYS).forEach(key => {
      localStorage.removeItem(key);
    });
  }, []);

  const [assets, setAssets] = useState<Asset[]>([]);
  const [summary, setSummary] = useState<PortfolioSummary | null>(null);
  const [allocation, setAllocation] = useState<AssetAllocation | null>(null);
  const [recommendations, setRecommendations] = useState<Recommendation[]>([]);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [nextUpdate, setNextUpdate] = useState<Date | null>(null);
  const [loading, setLoading] = useState({
    assets: true,
    summary: true,
    allocation: true,
    recommendations: true
  });
  const [errors, setErrors] = useState({
    assets: null as string | null,
    summary: null as string | null,
    allocation: null as string | null,
    recommendations: null as string | null
  });

  const fetchAssets = useCallback(async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/portfolio/assets');
      setAssets(response.data.assets);
      setLastUpdate(new Date());
      
      if (response.data.update_status.next_update_attempt) {
        setNextUpdate(new Date(response.data.update_status.next_update_attempt));
      }
      
      setErrors(prev => ({ ...prev, assets: null }));
    } catch (err) {
      setErrors(prev => ({ ...prev, assets: 'Failed to fetch portfolio assets' }));
      console.error('Error fetching assets:', err);
    } finally {
      setLoading(prev => ({ ...prev, assets: false }));
    }
  }, []);

  const fetchSummary = useCallback(async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/portfolio/summary');
      setSummary(response.data);
      setErrors(prev => ({ ...prev, summary: null }));
    } catch (err) {
      setErrors(prev => ({ ...prev, summary: 'Failed to fetch portfolio summary' }));
      console.error('Error fetching summary:', err);
    } finally {
      setLoading(prev => ({ ...prev, summary: false }));
    }
  }, []);

  const fetchAllocation = useCallback(async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/portfolio/allocation');
      setAllocation(response.data);
      setErrors(prev => ({ ...prev, allocation: null }));
    } catch (err) {
      setErrors(prev => ({ ...prev, allocation: 'Failed to fetch portfolio allocation' }));
      console.error('Error fetching allocation:', err);
    } finally {
      setLoading(prev => ({ ...prev, allocation: false }));
    }
  }, []);

  const fetchRecommendations = useCallback(async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/portfolio/recommendations');
      setRecommendations(response.data.recommendations);
      setErrors(prev => ({ ...prev, recommendations: null }));
    } catch (err) {
      setErrors(prev => ({ ...prev, recommendations: 'Failed to fetch recommendations' }));
      console.error('Error fetching recommendations:', err);
    } finally {
      setLoading(prev => ({ ...prev, recommendations: false }));
    }
  }, []);

  const refreshPrices = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/portfolio/update-prices');
      
      if (response.data.cache_info && response.data.cache_info.next_update) {
        setNextUpdate(new Date(response.data.cache_info.next_update));
      }
      
      // Refresh all data after price update
      await Promise.all([
        fetchAssets(),
        fetchSummary(),
        fetchAllocation()
      ]);
    } catch (err) {
      console.error('Error refreshing prices:', err);
      throw err;
    }
  };

  // Initial data fetch
  useEffect(() => {
    Promise.all([
      fetchAssets(),
      fetchSummary(),
      fetchAllocation(),
      fetchRecommendations()
    ]);

    // Set up refresh intervals
    const intervals = [
      setInterval(fetchAssets, CACHE_EXPIRY.ASSETS),
      setInterval(fetchSummary, CACHE_EXPIRY.SUMMARY),
      setInterval(fetchAllocation, CACHE_EXPIRY.ALLOCATION),
      setInterval(fetchRecommendations, CACHE_EXPIRY.RECOMMENDATIONS)
    ];

    return () => intervals.forEach(clearInterval);
  }, [fetchAssets, fetchSummary, fetchAllocation, fetchRecommendations]);

  return (
    <PortfolioContext.Provider
      value={{
        assets,
        summary,
        allocation,
        recommendations,
        loading,
        errors,
        refreshAssets: fetchAssets,
        refreshPrices,
        refreshRecommendations: fetchRecommendations,
        lastUpdate,
        nextUpdate
      }}
    >
      {children}
    </PortfolioContext.Provider>
  );
};

export const usePortfolio = () => {
  const context = useContext(PortfolioContext);
  if (context === undefined) {
    throw new Error('usePortfolio must be used within a PortfolioProvider');
  }
  return context;
}; 