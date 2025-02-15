import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
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
  FormControl,
  InputLabel,
  Select,
  MenuItem,
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
  model?: string;
}

interface ResearchResponse {
  answer: string;
  sources: {
    query: string;
    portfolio?: {
      summary: {
        total_value: number;
        total_cost: number;
        total_gain_loss: number;
        gain_loss_percentage: number;
        asset_count: number;
        type_allocation: Record<string, number>;
      };
      assets: Array<{
        symbol: string;
        name: string;
        type: string;
        quantity: number;
        current_price: number;
        purchase_price: number;
        current_value: number;
        weight: number;
        gain_loss: number;
        gain_loss_percentage: number;
      }>;
    };
    asset?: {
      symbol: string;
      name: string;
      type: string;
      quantity: number;
      current_price: number;
      purchase_price: number;
      current_value: number;
      portfolio_weight: number;
      gain_loss: number;
      gain_loss_percentage: number;
    };
    web_data?: any;
    timestamp: string;
  };
  model_used: string;
}

interface SuggestedQuery {
  text: string;
  description: string;
}

interface ModelInfo {
  id: string;
  name: string;
  description: string;
  is_available: boolean;
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
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const messagesEndRef = useRef<null | HTMLDivElement>(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const [isStreaming, setIsStreaming] = useState(false);
  const [currentStreamedMessage, setCurrentStreamedMessage] = useState('');

  // Fetch available models
  useEffect(() => {
    const fetchModels = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/models');
        setModels(response.data);
        // Set first available model as default
        const availableModel = response.data.find((m: ModelInfo) => m.is_available);
        if (availableModel) {
          setSelectedModel(availableModel.id);
        }
      } catch (err) {
        console.error('Error fetching models:', err);
      }
    };
    fetchModels();
  }, []);

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
    const scrollToBottom = () => {
      if (messagesEndRef.current) {
        messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
      }
    };

    // Scroll when messages change
    scrollToBottom();

    // Set up an interval to scroll while streaming
    let scrollInterval: NodeJS.Timeout | null = null;
    if (isStreaming) {
      scrollInterval = setInterval(scrollToBottom, 100); // Scroll every 100ms while streaming
    }

    // Cleanup interval
    return () => {
      if (scrollInterval) {
        clearInterval(scrollInterval);
      }
    };
  }, [messages, isStreaming, currentStreamedMessage]); // Add currentStreamedMessage as dependency

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
    setIsStreaming(true);
    setCurrentStreamedMessage('');

    try {
      let accumulatedContent = '';
      let modelUsed = '';

      const response = await axios.post<ResearchResponse>(
        'http://localhost:8000/api/research/query',
        {
          query: query,
          context: {
            include_portfolio: !selectedAsset,
            symbol: selectedAsset?.symbol || null,
          },
          should_use_web: useWeb,
          model: selectedModel,
        },
        {
          responseType: 'text',
          onDownloadProgress: (progressEvent) => {
            const data = progressEvent.event.target.responseText;
            try {
              // Split the response by newlines to handle streaming JSON
              const lines = data.split('\n');
              for (const line of lines) {
                if (line.trim()) {
                  const parsedData = JSON.parse(line);
                  if (parsedData.answer) {
                    accumulatedContent += parsedData.answer;
                    setCurrentStreamedMessage(accumulatedContent);
                    if (parsedData.model_used) {
                      modelUsed = parsedData.model_used;
                    }
                  }
                }
              }
            } catch (e) {
              // Ignore JSON parse errors for incomplete chunks
              console.debug('Parse error for chunk:', e);
            }
          }
        }
      );

      // Create the final message once streaming is complete
      const assistantMessage: Message = {
        role: 'assistant',
        content: accumulatedContent,
        timestamp: new Date().toISOString(),
        model: modelUsed,
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (err) {
      console.error('Error sending query:', err);
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error processing your request. Please try again.',
        timestamp: new Date().toISOString(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsStreaming(false);
      setCurrentStreamedMessage('');
      setShowSuggestions(true);
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
              selected={selectedAsset === null}
              onClick={() => handleAssetSelect(null)}
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
                  secondary={`${asset.name} - $${asset.current_value.toLocaleString()}`}
                />
              </ListItemButton>
            ))}
          </List>
        </Paper>
      </Grid>

      {/* Chat Interface */}
      <Grid item xs={12} md={9}>
        <Paper sx={{ p: 2, height: 'calc(100vh - 140px)', display: 'flex', flexDirection: 'column' }}>
          {/* Model Selection and Web Data Toggle */}
          <Box sx={{ mb: 2, display: 'flex', gap: 2, alignItems: 'center' }}>
            <FormControl sx={{ minWidth: 200 }}>
              <InputLabel>AI Model</InputLabel>
              <Select
                value={selectedModel || ''}
                label="AI Model"
                onChange={(e) => setSelectedModel(e.target.value)}
              >
                {models.map((model) => (
                  <MenuItem
                    key={model.id}
                    value={model.id}
                    disabled={!model.is_available}
                  >
                    <Box>
                      <Typography variant="body1">
                        {model.name}
                        {!model.is_available && ' (Unavailable)'}
                      </Typography>
                      <Typography variant="caption" color="text.secondary">
                        {model.description}
                      </Typography>
                    </Box>
                  </MenuItem>
                ))}
              </Select>
            </FormControl>
            <FormControlLabel
              control={
                <Switch
                  checked={useWeb}
                  onChange={(e) => setUseWeb(e.target.checked)}
                />
              }
              label="Include Web Data"
            />
          </Box>

          {/* Messages Area */}
          <Box 
            sx={{ 
              flexGrow: 1, 
              overflow: 'auto', 
              mb: 2,
              scrollBehavior: 'smooth' // Add smooth scrolling
            }}
          >
            <List>
              {messages.map((message, index) => (
                <ListItem key={index} sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      {message.role === 'assistant' ? 'AI Assistant' : 'You'}
                    </Typography>
                    {message.model && (
                      <Chip
                        label={models.find(m => m.id === message.model)?.name || message.model}
                        size="small"
                        variant="outlined"
                      />
                    )}
                    <Typography variant="caption" color="text.secondary">
                      {new Date(message.timestamp).toLocaleTimeString()}
                    </Typography>
                  </Box>
                  <Box
                    sx={{
                      backgroundColor: message.role === 'assistant' ? 'action.hover' : 'transparent',
                      p: message.role === 'assistant' ? 1 : 0,
                      borderRadius: 1,
                      width: '100%',
                    }}
                  >
                    <ReactMarkdown>{message.content}</ReactMarkdown>
                  </Box>
                </ListItem>
              ))}
              {isStreaming && (
                <ListItem sx={{ flexDirection: 'column', alignItems: 'flex-start' }}>
                  <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography variant="caption" color="text.secondary">
                      AI Assistant
                    </Typography>
                    <CircularProgress size={16} />
                  </Box>
                  <Box
                    sx={{
                      backgroundColor: 'action.hover',
                      p: 1,
                      borderRadius: 1,
                      width: '100%',
                    }}
                  >
                    <ReactMarkdown>{currentStreamedMessage}</ReactMarkdown>
                  </Box>
                </ListItem>
              )}
              <div ref={messagesEndRef} style={{ height: '1px' }} /> {/* Add minimal height to prevent layout shifts */}
            </List>
          </Box>

          {/* Suggested Queries */}
          {showSuggestions && (
            <Box sx={{ mb: 2 }}>
              <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                Suggested Questions
              </Typography>
              <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                {(selectedAsset ? assetSuggestedQueries : portfolioSuggestedQueries)
                  .map((suggestion, index) => (
                    <Chip
                      key={index}
                      label={suggestion.text}
                      onClick={() => handleSuggestedQuery(suggestion.text)}
                      sx={{ mb: 1 }}
                    />
                  ))}
              </Box>
            </Box>
          )}

          {/* Input Area */}
          <Box sx={{ display: 'flex', gap: 1 }}>
            <TextField
              fullWidth
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSubmit()}
              placeholder="Ask about your portfolio or selected asset..."
              disabled={loading}
            />
            <Button
              variant="contained"
              onClick={handleSubmit}
              disabled={loading || !query.trim()}
              endIcon={loading ? <CircularProgress size={20} /> : <SendIcon />}
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