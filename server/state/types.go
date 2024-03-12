package state

import (
	"slices"
)

type Model struct {
	Name           string `json:"name"`
	Config         ConfigV2
	ShortName      string
	ModelPath      string
	ParentModel    string
	AdapterPaths   []string
	ProjectorPaths []string
	Template       string
	System         string
	License        []string
	Digest         string
	Size           int64
	Options        map[string]interface{}
	Messages       []Message
}

func (m *Model) IsEmbedding() bool {
	return slices.Contains(m.Config.ModelFamilies, "bert") || slices.Contains(m.Config.ModelFamilies, "nomic-bert")
}

type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ConfigV2 struct {
	ModelFormat   string   `json:"model_format"`
	ModelFamily   string   `json:"model_family"`
	ModelFamilies []string `json:"model_families"`
	ModelType     string   `json:"model_type"`
	FileType      string   `json:"file_type"`

	// required by spec
	Architecture string `json:"architecture"`
	OS           string `json:"os"`
	RootFS       RootFS `json:"rootfs"`
}

func (c *ConfigV2) SetModelFormat(format string) {
	if c.ModelFormat == "" {
		c.ModelFormat = format
	}
}

func (c *ConfigV2) SetModelFamily(families ...string) {
	for _, family := range families {
		if c.ModelFamily == "" {
			c.ModelFamily = family
		}

		if !slices.Contains(c.ModelFamilies, family) {
			c.ModelFamilies = append(c.ModelFamilies, family)
		}
	}
}

func (c *ConfigV2) SetModelType(modelType string) {
	if c.ModelType == "" {
		c.ModelType = modelType
	}
}

func (c *ConfigV2) SetFileType(fileType string) {
	if c.FileType == "" {
		c.FileType = fileType
	}
}

type RootFS struct {
	Type    string   `json:"type"`
	DiffIDs []string `json:"diff_ids"`
}
