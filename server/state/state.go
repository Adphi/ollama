package state

import (
	"errors"
	"fmt"
	"log/slog"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/jmorganca/ollama/api"
	"github.com/jmorganca/ollama/llm"
)

var loaded State

type State struct {
	runner llm.LLM

	expireAt    time.Time
	expireTimer *time.Timer
	tmu         sync.RWMutex

	*Model
	*api.Options
	mu sync.RWMutex
}

func (s *State) Runner() llm.LLM {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.runner
}

func (s *State) ResetExpireTimer(sessionDuration time.Duration) {
	s.tmu.Lock()
	defer s.tmu.Unlock()
	s.expireAt = time.Now().Add(sessionDuration)
	if s.expireTimer != nil {
		s.expireTimer.Reset(sessionDuration)
	}
}

func (s *State) Close() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.tmu.Lock()
	defer s.tmu.Unlock()
	if s.runner != nil {
		s.runner.Close()
	}
	s.runner = nil
	s.Model = nil
	s.Options = nil
	s.expireAt = time.Time{}
	if s.expireTimer != nil {
		s.expireTimer.Stop()
	}
}

// Load a model into memory if it is not already loaded, it is up to the caller to lock loaded.mu before calling this function
func Load(model *Model, opts api.Options, sessionDuration time.Duration) (*State, error) {
	loaded.mu.RLock()
	needLoad := loaded.runner == nil || // is there a model loaded?
		loaded.ModelPath != model.ModelPath || // has the base model changed?
		!reflect.DeepEqual(loaded.AdapterPaths, model.AdapterPaths) || // have the adapters changed?
		!reflect.DeepEqual(loaded.Options.Runner, opts.Runner) // have the runner options changed?
	loaded.mu.RUnlock()

	if needLoad {
		if err := load(model, opts); err != nil {
			return nil, err
		}
	}

	loaded.tmu.Lock()
	defer loaded.tmu.Unlock()
	loaded.expireAt = time.Now().Add(sessionDuration)

	if loaded.expireTimer == nil {
		loaded.expireTimer = time.AfterFunc(sessionDuration, func() {
			loaded.mu.Lock()
			defer loaded.mu.Unlock()

			if time.Now().Before(loaded.expireAt) {
				return
			}

			if loaded.runner != nil {
				loaded.runner.Close()
			}

			loaded.runner = nil
			loaded.Model = nil
			loaded.Options = nil
		})
	}

	loaded.expireTimer.Reset(sessionDuration)
	return &loaded, nil
}

func load(model *Model, opts api.Options) error {
	loaded.mu.Lock()
	defer loaded.mu.Unlock()
	if loaded.runner != nil {
		slog.Info("changing loaded model")
		loaded.runner.Close()
		loaded.runner = nil
		loaded.Model = nil
		loaded.Options = nil
	}

	llmRunner, err := llm.New(model.ModelPath, model.AdapterPaths, model.ProjectorPaths, opts)
	if err != nil {
		// some older models are not compatible with newer versions of llama.cpp
		// show a generalized compatibility error until there is a better way to
		// check for model compatibility
		if errors.Is(llm.ErrUnsupportedFormat, err) || strings.Contains(err.Error(), "failed to load model") {
			err = fmt.Errorf("%v: this model may be incompatible with your version of Ollama. If you previously pulled this model, try updating it by running `ollama pull %s`", err, model.ShortName)
		}

		return err
	}

	loaded.Model = model
	loaded.runner = newSyncLLM(llmRunner)
	loaded.Options = &opts
	return nil
}

func Close() {
	loaded.Close()
}
