package state

import (
	"context"
	"errors"
	"sync"

	"github.com/jmorganca/ollama/llm"
)

var _ llm.LLM = (*syncLLM)(nil)

var ErrLLMClosed = errors.New("llm closed")

type syncLLM struct {
	mu     sync.RWMutex
	llm    llm.LLM
	closed bool
}

func newSyncLLM(llm llm.LLM) *syncLLM {
	return &syncLLM{
		llm: llm,
	}
}

func (s *syncLLM) Predict(ctx context.Context, opts llm.PredictOpts, f func(llm.PredictResult)) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return ErrLLMClosed
	}
	return s.llm.Predict(ctx, opts, f)
}

func (s *syncLLM) Embedding(ctx context.Context, s2 string) ([]float64, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil, ErrLLMClosed
	}
	return s.llm.Embedding(ctx, s2)
}

func (s *syncLLM) Encode(ctx context.Context, s2 string) ([]int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return nil, ErrLLMClosed
	}
	return s.llm.Encode(ctx, s2)
}

func (s *syncLLM) Decode(ctx context.Context, i []int) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return "", ErrLLMClosed
	}
	return s.llm.Decode(ctx, i)
}

func (s *syncLLM) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.llm.Close()
	s.closed = true
}
