job "rids_staging_notify_ping" {
  datacenters = ["dc1"]

  type = "batch"

  periodic {
    cron             = "* * * * *"
    prohibit_overlap = true
    time_zone        = "America/New_York"
  }

  group "default" {
    vault {
      policies = ["rids_staging"]
    }

    restart {
      attempts = 90
      delay    = "5s"
      interval = "24h"
      mode     = "fail"
    }

    task "ping" {
      driver = "docker"
      config = {
        command = "rids.ping_notify"
        image = "quay.io/pennsignals/rids:1.2"
        force_pull = true

        volumes = [
          "/share/models/rids/:/tmp/model:ro",
          "/share/models/rids/:/tmp/temporary_output:ro"
        ]
      }
      env {
        CONFIGURATION="/local/notify_configuration.yml"
        ANTIBIOTICS_SNOMED="/local/snomedid_description.txt"
        ANTIBIOTICS_SNOMED_NORM="/local/snomedid_description_norm_map.txt"
        ANTIBIOTICS_BRAND_GENERIC="/local/brands_and_generic.csv"
      }
      resources {
        cpu    = 4096
        memory = 2048
      }
      template {
        data =<<EOH
{{key "rids_staging/notify_configuration.yml"}}
EOH
        destination = "/local/notify_configuration.yml"
      }
      template {
        data =<<EOH
{{key "rids_staging/snomedid_description.txt"}}
EOH
        destination = "/local/snomedid_description.txt"
      }
      template {
        data =<<EOH
{{key "rids_staging/snomedid_description_norm_map.txt"}}
EOH
        destination = "/local/snomedid_description_norm_map.txt"
      }
      template {
        data =<<EOH
{{key "rids_staging/brands_and_generic.csv"}}
EOH
        destination = "/local/brands_and_generic.csv"
      }
      template {
        data =<<EOH
PREDICTION_INPUT_URI="{{with secret "secret/mongo/rids_staging/prediction_input_uri"}}{{.Data.value}}{{end}}"
NOTIFY_OUTPUT_URI="{{with secret "secret/mongo/rids_staging/notify_output_uri"}}{{.Data.value}}{{end}}"
VENT_INPUT_URI="{{with secret "secret/mongo/rids_staging/vent_input_uri"}}{{.Data.value}}{{end}}"
PENNCHARTX_CLIENT_SECRET="{{with secret "secret/pennchartx/rids_staging/client_secret"}}{{.Data.value}}{{end}}"
PENNCHARTX_PASSWORD="{{with secret "secret/pennchartx/rids_staging/password"}}{{.Data.value}}{{end}}"
PENNCHARTX_POSTMAN_TOKEN="{{with secret "secret/pennchartx/rids_staging/postman_token"}}{{.Data.value}}{{end}}"
REDCAP_TOKEN="{{with secret "secret/redcap/rids_staging/token"}}{{.Data.value}}{{end}}"
SLACK_STARTUP_URL="{{with secret "secret/slack/rids_staging/startup_url"}}{{.Data.value}}{{end}}"
SLACK_ALERT_HUP_URL="{{with secret "secret/slack/rids_staging/alert_hup_url"}}{{.Data.value}}{{end}}"
SLACK_ALERT_PMC_URL="{{with secret "secret/slack/rids_staging/alert_pmc_url"}}{{.Data.value}}{{end}}"
EOH
        destination = "/secrets/.env"
        env         = true
      }
    }
  }
}
