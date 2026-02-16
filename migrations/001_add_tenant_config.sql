-- Migration: Add tenant_config table for pipeline configuration
-- Date: 2026-02-16

CREATE TABLE IF NOT EXISTS tenant_config (
    tenant_id VARCHAR(100) PRIMARY KEY,
    pipeline_steps JSONB NOT NULL,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for fast lookups
CREATE INDEX IF NOT EXISTS idx_tenant_config_tenant_id ON tenant_config(tenant_id);

-- Trigger to update updated_at on changes
CREATE OR REPLACE FUNCTION update_tenant_config_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS tenant_config_updated_at ON tenant_config;
CREATE TRIGGER tenant_config_updated_at
    BEFORE UPDATE ON tenant_config
    FOR EACH ROW
    EXECUTE FUNCTION update_tenant_config_updated_at();

-- Example data for testing/demo
INSERT INTO tenant_config (tenant_id, pipeline_steps, settings) VALUES
(
    'cultivos-demo',
    '["segmentation", "segment_filter", "detection", "size_calculator", "species_distributor"]',
    '{"num_bands": 4, "species": ["Tomato", "Pepper", "Lettuce"], "segment_filter_type": "largest_claro", "image_height": 4000}'
) ON CONFLICT (tenant_id) DO NOTHING;
