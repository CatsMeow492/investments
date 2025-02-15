import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  CircularProgress,
  List,
  ListItem,
  ListItemText,
  ListItemButton,
  Divider,
  Switch,
  FormControlLabel,
  Chip,
} from '@mui/material';
import { Send as SendIcon, Search as SearchIcon } from '@mui/icons-material';

interface Asset {
  symbol: string;
  name: string;
  type: string;
  quantity: number;
  current_price: number;
  current_value: number;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface ResearchResponse {
  answer: string;
  sources: string[];
  context_used: any;
}

interface SuggestedQuery {
  text: string;
  description: string;
}

const portfolioSuggestedQueries: SuggestedQuery[] = [
  {
    text: "What is my portfolio's sector allocation?",
    description: "Analyze distribution across different sectors"
  },
  {
    text: "Which investments have the highest potential for growth?",
    description: "Analyze growth potential based on current market conditions"
  },
  {
    text: "What are the biggest risks in my portfolio?",
    description: "Identify concentration risks and market vulnerabilities"
  },
  {
    text: "How can I better diversify my portfolio?",
    description: "Get suggestions for improving diversification"
  },
  {
    text: "What are the latest news affecting my portfolio?",
    description: "Get relevant news about your investments"
  }
];

const assetSuggestedQueries: SuggestedQuery[] = [
  {
    text: "What's the recent performance and outlook?",
    description: "Get performance analysis and future outlook"
  },
  {
    text: "What are the key risks to watch?",
    description: "Understand potential risks and challenges"
  },
  {
    text: "How does this compare to competitors?",
    description: "Competitive analysis and market position"
  },
  {
    text: "What are analysts saying?",
    description: "Recent analyst ratings and price targets"
  }
];

const Research: React.FC = () => {
  const [assets, setAssets] = useState<Asset[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedAsset, setSelectedAsset] = useState<Asset | null>(null);
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [useWeb, setUseWeb] = useState(true);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);

  // Fetch portfolio assets
  useEffect(() => {
    const fetchAssets = async () => {
      try {
        setLoading(true);
        const response = await axios.get('http://localhost:8000/api/portfolio/assets');
        setAssets(response.data);
      } catch (err) {
        setError('Failed to fetch portfolio assets');
        console.error('Error fetching assets:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchAssets();
  }, []);

  // Auto-scroll to bottom of chat
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSuggestedQuery = (queryText: string) => {
    setQuery(queryText);
    setShowSuggestions(false);
  };

  const getInitialMessage = (asset: Asset | null) => {
    if (asset) {
      return {
        role: 'assistant' as const,
        content: `I'm ready to help you research ${asset.name} (${asset.symbol}). Here are some things you might want to know:
        
• Current position: ${asset.quantity.toFixed(4)} shares at $${asset.current_price.toFixed(2)}
• Total value: $${asset.current_value.toLocaleString()}

What would you like to know about this investment?`,
        timestamp: new Date().toISOString(),
      };
    } else {
      const totalValue = assets.reduce((sum, asset) => sum + asset.current_value, 0);
      const assetCount = assets.length;
      return {
        role: 'assistant' as const,
        content: `I'm ready to help you research your entire portfolio. Here's a quick overview:

• Total portfolio value: $${totalValue.toLocaleString()}
• Number of investments: ${assetCount}
• Top holdings: ${assets.slice(0, 3).map(a => a.symbol).join(', ')}

I can help you analyze your portfolio's composition, risk factors, performance, and suggest optimization strategies. What would you like to know?`,
        timestamp: new Date().toISOString(),
      };
    }
  };

  const handleAssetSelect = (asset: Asset | null) => {
    setSelectedAsset(asset);
    setShowSuggestions(true);
    setMessages([getInitialMessage(asset)]);
  };

  const handleSubmit = async () => {
    if (!query.trim()) return;

    const userMessage: Message = {
      role: 'user',
      content: query,
      timestamp: new Date().toISOString(),
    };

    setMessages(prev => [...prev, userMessage]);
    setQuery('');

    try {
      setLoading(true);
      const response = await axios.post<ResearchResponse>('http://localhost:8000/api/research/query', {
        query: query,
        context: {
          include_portfolio: !selectedAsset,
          symbol: selectedAsset?.symbol || null,
        },
        should_use_web: useWeb,
      });

      const assistantMessage: Message = {
        role: 'assistant',
        content: response.data.answer,
        timestamp: new Date().toISOString(),
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending query:', err);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request.',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Grid container spacing={3}>
      {/* Asset Selection */}
      <Grid item xs={12} md={3}>
        <Paper sx={{ p: 2, height: 'calc(100vh - 140px)', overflow: 'auto' }}>
          <Typography variant="h6" gutterBottom>
            Portfolio Assets
          </Typography>
          <List>
            <ListItemButton
              selected={!selectedAsset}
              onClick={() => {
                setSelectedAsset(null);
                setMessages([{
                  role: 'assistant',
                  content: "I'm ready to help you research your entire portfolio. What would you like to know?",
                  timestamp: new Date().toISOString(),
                }]);
              }}
            >
              <ListItemText primary="Entire Portfolio" />
            </ListItemButton>
            <Divider />
            {assets.map((asset) => (
              <ListItemButton
                key={asset.symbol}
                selected={selectedAsset?.symbol === asset.symbol}
                onClick={() => handleAssetSelect(asset)}
              >
                <ListItemText
                  primary={asset.symbol}
                  secondary={asset.name}
                />
                <Chip
                  label={`$${asset.current_value.toLocaleString()}`}
                  size="small"
                  color="primary"
                />
              </ListItemButton>
            ))}
          </List>
        </Paper>
      </Grid>

      {/* Chat Interface */}
      <Grid item xs={12} md={9}>
        <Paper sx={{ p: 2, height: 'calc(100vh - 140px)', display: 'flex', flexDirection: 'column' }}>
          <Box sx={{ mb: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Typography variant="h6">
              {selectedAsset ? `Research: ${selectedAsset.name} (${selectedAsset.symbol})` : 'Portfolio Research'}
            </Typography>
            <FormControlLabel
              control={
                <Switch
                  checked={useWeb}
                  onChange={(e) => setUseWeb(e.target.checked)}
                  color="primary"
                />
              }
              label="Use Web Data"
            />
          </Box>

          {/* Messages */}
          <Box sx={{ flexGrow: 1, overflow: 'auto', mb: 2 }}>
            <List>
              {messages.map((message, index) => (
                <ListItem
                  key={index}
                  sx={{
                    backgroundColor: message.role === 'assistant' ? 'action.hover' : 'transparent',
                    borderRadius: 1,
                    mb: 1,
                  }}
                >
                  <ListItemText
                    primary={message.content}
                    secondary={new Date(message.timestamp).toLocaleString()}
                  />
                </ListItem>
              ))}
              {showSuggestions && messages.length === 1 && (
                <ListItem>
                  <Box sx={{ width: '100%' }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Suggested Questions:
                    </Typography>
                    <Grid container spacing={1}>
                      {(selectedAsset ? assetSuggestedQueries : portfolioSuggestedQueries).map((query, index) => (
                        <Grid item xs={12} key={index}>
                          <Button
                            variant="outlined"
                            size="small"
                            onClick={() => handleSuggestedQuery(query.text)}
                            sx={{ justifyContent: 'flex-start', textAlign: 'left', width: '100%' }}
                          >
                            <Box>
                              <Typography variant="body2">{query.text}</Typography>
                              <Typography variant="caption" color="text.secondary">
                                {query.description}
                              </Typography>
                            </Box>
                          </Button>
                        </Grid>
                      ))}
                    </Grid>
                  </Box>
                </ListItem>
              )}
            </List>
            <div ref={messagesEndRef} />
          </Box>

          {/* Input */}
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              variant="outlined"
              placeholder="Ask about your investments..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
              disabled={loading}
            />
            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={loading || !query.trim()}
              startIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
            >
              Send
            </Button>
          </Box>
        </Paper>
      </Grid>
    </Grid>
  );
};

export default Research; 