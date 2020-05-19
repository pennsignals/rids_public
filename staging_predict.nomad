job "rids_staging_predict" {
  datacenters = ["dc1"]

  type="batch"

  periodic = {
    cron             = "*/15 * * * *"
    prohibit_overlap = true
    time_zone        = "America/New_York"
  }

  group "default" {
    vault {
      policies = ["rids_staging"]
    }

    restart {
      attempts = 24
      delay    = "5m"
      interval = "24h"
      mode     = "fail"
    }

    task "prediction" {
      driver = "docker"
      config = {
        command = "rids.predict"
        image = "quay.io/pennsignals/rids:1.2"
        force_pull = true

        volumes = [
          "/share/models/rids/:/tmp/model:ro",
          "/share/models/rids/:/tmp/temporary_output:rw"
        ]
      }
      env {
        CONFIGURATION="/local/predict_configuration.yml"
        ELIXHAUSER="/local/elixhauser_map.tsv"
        CHARLSON="/local/charlson_map.tsv"
        ORDERS="/local/orders_map.tsv"
        VITALS="/local/vitals_map.tsv"
      }
      resources {
        cpu    = 4096
        memory = 2048
      }
      template {
        data =<<EOH
{{key "rids_staging/predict_configuration.yml"}}
EOH
        destination = "/local/predict_configuration.yml"
      }
      template {
        data =<<EOH
{{key "rids_staging/elixhauser_map.tsv"}}
EOH
        destination = "/local/elixhauser_map.tsv"
      }
      template {
        data =<<EOH
{{key "rids_staging/charlson_map.tsv"}}
EOH
        destination = "/local/charlson_map.tsv"
      }
      template {
        data =<<EOH
{{key "rids_staging/orders_map.tsv"}}
EOH
        destination = "/local/orders_map.tsv"
      }
      template {
        data =<<EOH
{{key "rids_staging/vitals_map.tsv"}}
EOH
        destination = "/local/vitals_map.tsv"
      }

      template {
        data =<<EOH
OUTPUT_URI="{{with secret "secret/mongo/rids_staging/output_uri"}}{{.Data.value}}{{end}}"
CLARITY_INPUT_URI="{{with secret "secret/mssql/rids_staging/clarity_input_uri"}}{{.Data.value}}{{end}}"
VENT_INPUT_URI="{{with secret "secret/mongo/rids_staging/vent_input_uri"}}{{.Data.value}}{{end}}"
PS1_INPUT_URI="{{with secret "secret/mongo/rids_staging/ps1_input_uri"}}{{.Data.value}}{{end}}"
EOH
        destination = "/secrets/.env"
        env         = true
      }
    }
  }
}
